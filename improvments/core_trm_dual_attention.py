import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder

from core_ave import COREave


class TransNetDualAttention(nn.Module):
    """
    Enhanced Transformer Network with Dual Attention Mechanism.
    Combines global session context with recent-item emphasis.
    """
    def __init__(self, config, dataset):
        super().__init__()

        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']

        self.position_embedding = nn.Embedding(dataset.field2seqlen['item_id_list'], self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Global context attention (similar to original CORE-trm)
        self.global_attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
        )

        # Recent-item attention mechanism (inspired by STAMP/NARM)
        # Query-based attention using last item as query
        self.recent_attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
        )

        # Gating mechanism to balance global and recent context
        # Takes concatenation of both representations and produces gate weights
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 2),  # 2 outputs: weight for global, weight for recent
            nn.Softmax(dim=-1)
        )

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        batch_size = item_seq.size(0)
        seq_len = item_seq.size(1)
        mask = item_seq.gt(0)

        # Position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(position_ids)

        # Transformer encoding
        input_emb = item_emb + pos_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        # === Global Context Attention (Original CORE-trm style) ===
        combined_input = torch.cat((output, pos_emb), dim=-1)
        alpha_global = self.global_attention(combined_input).to(torch.double)
        alpha_global = torch.where(mask.unsqueeze(-1), alpha_global, -9e15)
        alpha_global = torch.softmax(alpha_global, dim=1, dtype=torch.float)

        # === Recent-Item Attention (STAMP/NARM inspired) ===
        # Extract last valid item embedding for each sequence in batch
        seq_lengths = mask.sum(dim=1)  # [batch_size]
        last_indices = (seq_lengths - 1).clamp(min=0)  # [batch_size]
        
        # Get last item embedding from transformer output
        last_item_emb = output[torch.arange(batch_size, device=output.device), last_indices]  # [batch_size, hidden_size]
        
        # Use last item as query to compute attention over all items
        # Expand last_item_emb to match sequence dimension
        last_item_expanded = last_item_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Compute similarity between last item and all items in session
        recent_query = torch.cat((output, last_item_expanded), dim=-1)  # [batch_size, seq_len, hidden_size*2]
        alpha_recent = self.recent_attention(recent_query).to(torch.double)
        alpha_recent = torch.where(mask.unsqueeze(-1), alpha_recent, -9e15)
        alpha_recent = torch.softmax(alpha_recent, dim=1, dtype=torch.float)

        return alpha_global, alpha_recent

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class COREtrmDualAttention(COREave):
    """
    CORE-trm with Dual Attention Mechanism.
    
    Combines global session context with explicit recent-item emphasis,
    inspired by STAMP and NARM architectures that balance long-term 
    session memory with short-term recent intent.
    """
    def __init__(self, config, dataset):
        super(COREtrmDualAttention, self).__init__(config, dataset)
        self.net = TransNetDualAttention(config, dataset)
        
        # Fusion parameters for combining global and recent representations
        # Can be simple weighted sum or learnable gating
        self.use_gating = config['use_dual_gating'] if 'use_dual_gating' in config else True
        
        if self.use_gating:
            # Learnable gate to balance global vs recent context
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, 2),
                nn.Softmax(dim=-1)
            )
        else:
            # Simple fixed weights (can be tuned as hyperparameters)
            self.global_weight = config['global_weight'] if 'global_weight' in config else 0.6
            self.recent_weight = config['recent_weight'] if 'recent_weight' in config else 0.4

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)
        
        # Get dual attention weights from transformer network
        alpha_global, alpha_recent = self.net(item_seq, x_clean)
        
        # Compute global context representation
        global_output = torch.sum(alpha_global * x_noisy, dim=1)
        
        # Compute recent-interest representation
        recent_output = torch.sum(alpha_recent * x_noisy, dim=1)
        
        # === Fusion Strategy ===
        if self.use_gating:
            # Learnable gating based on both representations
            fusion_input = torch.cat([global_output, recent_output], dim=-1)
            gate_weights = self.fusion_gate(fusion_input)  # [batch_size, 2]
            
            # Apply gates
            trm_output = gate_weights[:, 0:1] * global_output + gate_weights[:, 1:2] * recent_output
        else:
            # Fixed weighted combination
            trm_output = self.global_weight * global_output + self.recent_weight * recent_output
        
        # === Residual Connection with Average Pooling ===
        # Preserve CORE's representation consistency via residual
        alpha_ave = self.ave_net(item_seq)
        ave_output = torch.sum(alpha_ave * x_noisy, dim=1)
        
        # Combine dual-attention output with average baseline
        seq_output = trm_output + ave_output
        
        # Normalize for cosine similarity computation
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
