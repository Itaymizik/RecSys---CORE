import torch
import torch.nn as nn
import torch.nn.functional as F
from core_ave import COREave

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, inner_size):
        super().__init__()
        # SwiGLU typically has 2 linear projections for the gate and value, then 1 for output
        # Here we follow LLaMA style: (Swish(W_g x) * (W_v x)) * W_o
        # But standard Transformer FeedForward is: Linear -> Act -> Linear
        # SwiGLU replaces the first Linear -> Act.
        # So we have: x -> (Linear_gate(x) * SiLU(Linear_val(x))) -> Linear_out(x)
        # Wait, LLaMA SwiGLU: FFN(x) = W2(Silu(W1(x)) * W3(x))
        self.w1 = nn.Linear(hidden_size, inner_size, bias=False) # Gate
        self.w2 = nn.Linear(hidden_size, inner_size, bias=False) # Value
        self.w3 = nn.Linear(inner_size, hidden_size, bias=False) # Output

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class ModernTransformerLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, inner_size, hidden_dropout_prob, attn_dropout_prob):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        
        self.ln1 = RMSNorm(hidden_size)
        self.attn_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        
        self.ln2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, inner_size)
        self.ffn_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x, attention_mask=None):
        # Pre-Norm Architecture
        residual = x
        x = self.ln1(x)
        
        # Self-Attention
        batch_size, seq_len, _ = x.size()
        q = self.attn_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
             # attention_mask shape: [batch_size, 1, 1, seq_len] or similar
             scores = scores + attention_mask
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.attn_o(context)
        x = residual + self.ffn_dropout(output)
        
        # Feed Forward with SwiGLU
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.ffn_dropout(x)
        
        return x

class ModernTransNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.initializer_range = config['initializer_range']
        
        self.position_embedding = nn.Embedding(dataset.field2seqlen['item_id_list'] + 1, self.hidden_size) # +1 buffer
        
        self.layers = nn.ModuleList([
            ModernTransformerLayer(
                self.hidden_size, 
                self.n_heads, 
                self.inner_size, 
                self.hidden_dropout_prob, 
                self.attn_dropout_prob
            ) for _ in range(self.n_layers)
        ])
        
        self.final_norm = RMSNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        # Modern MLP for alpha calculation (SwiGLU style or just simplified)
        # Using vanilla Sequential with GELU for the gate as in COREtrm is fine, 
        # but let's stick to simple Linear->Swish->Linear for consistency if we wanted, 
        # but here we just need a scalar.
        self.fn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(), # Modern activation
            nn.Linear(self.hidden_size, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [batch, 1, 1, seq_len]
        # Causal mask
        seq_len = item_seq.size(1)
        subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=item_seq.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)
        
        extended_attention_mask = extended_attention_mask & subsequent_mask
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)
        seq_len = item_seq.size(1)
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(position_ids)
        
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        attention_mask = self.get_attention_mask(item_seq)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        output = self.final_norm(x)
        last_output = output[:, -1, :] # Simplified: usually we want the last valid item, but padding creates issues.
        # But actually RecBole/CORE usually does gathering. 
        # COREtrm uses output[-1] from recbole encoder which returns all layers.
        # RecBole TransformerEncoder output: [batch, seq_len, hidden]
        
        # We need the last item's representation. 
        # Since we use 0-padding at end or beginning? RecBole usually pads at end.
        # COREtrm implementation:
        # trm_output = self.trm_encoder(...) [layers]
        # output = trm_output[-1] -> [batch, seq_len, hidden]
        # combined_input = cat(output, pos_emb)
        
        # So 'output' here corresponds to 'output' in COREtrm
        
        combined_input = torch.cat((output, pos_emb), dim=-1)
        alpha = self.fn(combined_input).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

class COREtrmModern(COREave):
    def __init__(self, config, dataset):
        super(COREtrmModern, self).__init__(config, dataset)
        self.net = ModernTransNet(config, dataset)
        
    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)
        
        alpha_trm = self.net(item_seq, x_clean)
        trm_output = torch.sum(alpha_trm * x_noisy, dim=1)
        
        alpha_ave = self.ave_net(item_seq)
        ave_output = torch.sum(alpha_ave * x_noisy, dim=1)
        
        seq_output = trm_output + ave_output
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
