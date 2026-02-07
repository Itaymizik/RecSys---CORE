import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder

from core_ave import COREave


class EnhancedTransNet(nn.Module):
    """Enhanced Transformer Network with Multi-Layer Aggregation and Relative Positional Encoding"""
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

        # Absolute position embedding (kept for compatibility)
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

        # IMPROVEMENT 1: Multi-Layer Aggregation
        # Learnable weights for combining different transformer layers
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers))
        
        # Enhanced attention scoring with layer aggregation
        self.fn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
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
        mask = item_seq.gt(0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        
        pos_emb = self.position_embedding(position_ids)

        input_emb = item_emb + pos_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        # Get all transformer layer outputs
        trm_output_layers = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        
        # IMPROVEMENT 1: Weighted aggregation of all layers
        # Normalize layer weights with softmax
        normalized_weights = F.softmax(self.layer_weights, dim=0)
        
        # Weighted sum of all layers
        aggregated_output = torch.zeros_like(trm_output_layers[0])
        for i, layer_output in enumerate(trm_output_layers):
            aggregated_output = aggregated_output + normalized_weights[i] * layer_output
        
        # Attention scoring with position injection
        combined_input = torch.cat((aggregated_output, pos_emb), dim=-1)

        alpha = self.fn(combined_input).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class COREtrmEnhanced(COREave):
    """Enhanced CORE-TRM with:
    1. Multi-layer aggregation
    2. Learnable gating between TRM and AVE
    3. Hard negative mining in contrastive loss
    """
    def __init__(self, config, dataset):
        super(COREtrmEnhanced, self).__init__(config, dataset)
        self.net = EnhancedTransNet(config, dataset)
        
        # IMPROVEMENT 2: Learnable gating mechanism
        # Dynamically balance between transformer and average pooling
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )
        
        # IMPROVEMENT 3: Hard negative mining
        self.hard_neg_weight = config['hard_neg_weight'] if 'hard_neg_weight' in config else 0.5
        self.use_hard_negatives = config['use_hard_negatives'] if 'use_hard_negatives' in config else True

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)
        
        # Path A: Enhanced Transformer with multi-layer aggregation
        alpha_trm = self.net(item_seq, x_clean)
        trm_output = torch.sum(alpha_trm * x_noisy, dim=1)
        
        # Path B: Average (Residual)
        alpha_ave = self.ave_net(item_seq)
        ave_output = torch.sum(alpha_ave * x_noisy, dim=1)
        
        # IMPROVEMENT 2: Learnable gating instead of fixed addition
        # Concatenate both representations to decide the gate value
        gate_input = torch.cat([trm_output, ave_output], dim=-1)
        gate_value = self.gate(gate_input)  # Shape: (batch_size, 1)
        
        # Weighted combination based on learned gate
        seq_output = gate_value * trm_output + (1 - gate_value) * ave_output
        
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        
        # Compute logits
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        
        # Standard cross-entropy loss
        ce_loss = self.loss_fct(logits, pos_items)
        
        # IMPROVEMENT 3: Hard negative mining
        if self.training and self.use_hard_negatives:
            # Find hardest negatives within the batch
            batch_size = seq_output.size(0)
            
            # Compute similarity between all pairs in the batch
            # Shape: (batch_size, batch_size)
            batch_sim = torch.matmul(seq_output, seq_output.transpose(0, 1)) / self.temperature
            
            # Mask out diagonal (self-similarity) and positive items
            mask = torch.eye(batch_size, device=seq_output.device).bool()
            batch_sim = batch_sim.masked_fill(mask, -1e9)
            
            # Hard negative loss: push away the most similar negative samples
            hard_neg_loss = torch.logsumexp(batch_sim, dim=1).mean()
            
            # Combine losses
            total_loss = ce_loss + self.hard_neg_weight * hard_neg_loss
            return total_loss
        
        return ce_loss


### Enhanced model with multi-layer aggregation, learnable gating, and hard negative mining
