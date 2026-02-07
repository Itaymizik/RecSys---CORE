import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder

from core_ave import COREave


class RelativePositionBias(nn.Module):
    """
    Relative Positional Encoding with learnable biases.
    Captures relative distances between items instead of absolute positions.
    """
    def __init__(self, n_heads, max_relative_position=32):
        super().__init__()
        self.n_heads = n_heads
        self.max_relative_position = max_relative_position
        
        # Learnable relative position biases
        # Range: [-max_relative_position, max_relative_position]
        self.relative_position_bias = nn.Embedding(
            2 * max_relative_position + 1, 
            n_heads
        )
        
    def forward(self, seq_len):
        """
        Generate relative position bias for self-attention.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            bias: [1, n_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.relative_position_bias.weight.device)
        
        # Compute relative positions: [seq_len, seq_len]
        # relative_positions[i, j] = j - i (how far j is from i)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clip to max range and shift to positive indices
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        relative_positions = relative_positions + self.max_relative_position
        
        # Get biases: [seq_len, seq_len, n_heads]
        bias = self.relative_position_bias(relative_positions)
        
        # Reshape to [1, n_heads, seq_len, seq_len] for attention
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        
        return bias


class ContextAwarePositionEncoding(nn.Module):
    """
    Context-Aware Position Encoding (inspired by CAPE).
    Generates position embeddings that adapt based on session context.
    """
    def __init__(self, hidden_size, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Base positional embedding (learnable)
        self.base_position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Context modulation network
        # Takes item embedding and generates position-specific weights
        self.context_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # Gate to modulate position embeddings
        )
        
        # Recency bias: give more weight to recent positions
        self.recency_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, item_emb, position_ids):
        """
        Generate context-aware position embeddings.
        
        Args:
            item_emb: [batch_size, seq_len, hidden_size]
            position_ids: [batch_size, seq_len]
            
        Returns:
            pos_emb: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = item_emb.shape
        
        # Get base positional embeddings
        base_pos_emb = self.base_position_embedding(position_ids)  # [B, L, H]
        
        # Generate context-dependent modulation gates
        # This allows position encoding to adapt based on item content
        context_gates = self.context_network(item_emb)  # [B, L, H]
        
        # Modulate position embeddings by context
        modulated_pos_emb = base_pos_emb * context_gates
        
        # Add recency bias: exponentially decay based on distance from end
        # Recent positions (near end) get higher weight
        max_pos = seq_len - 1
        position_indices = position_ids.float()
        recency_weights = torch.exp(
            -self.recency_weight * (max_pos - position_indices) / max_pos
        ).unsqueeze(-1)  # [B, L, 1]
        
        # Combine base, modulated, and recency-weighted embeddings
        pos_emb = modulated_pos_emb * recency_weights + base_pos_emb * (1 - recency_weights * 0.5)
        
        return pos_emb


class TransNetEnhancedPE(nn.Module):
    """
    Transformer Network with Enhanced Positional Encoding.
    Combines relative positional bias and context-aware position embeddings.
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
        
        max_seq_len = dataset.field2seqlen['item_id_list']
        
        # Enhanced positional encoding components
        self.use_relative_pe = config['use_relative_pe'] if 'use_relative_pe' in config else True
        self.use_context_aware_pe = config['use_context_aware_pe'] if 'use_context_aware_pe' in config else True
        
        # Context-aware position encoding (CAPE-inspired)
        if self.use_context_aware_pe:
            self.position_encoding = ContextAwarePositionEncoding(self.hidden_size, max_seq_len)
        else:
            # Fallback to standard learnable embeddings
            self.position_encoding = nn.Embedding(max_seq_len, self.hidden_size)
        
        # Relative position bias for self-attention
        if self.use_relative_pe:
            max_relative_pos = config['max_relative_position'] if 'max_relative_position' in config else 32
            self.relative_position_bias = RelativePositionBias(self.n_heads, max_relative_pos)
        
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

        # Attention aggregation network (with position awareness)
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
        
        # Add relative position bias if enabled
        if self.use_relative_pe:
            seq_len = item_seq.size(1)
            rel_pos_bias = self.relative_position_bias(seq_len)  # [1, n_heads, seq_len, seq_len]
            # Expand to batch size
            rel_pos_bias = rel_pos_bias.expand(item_seq.size(0), -1, -1, -1)
            extended_attention_mask = extended_attention_mask + rel_pos_bias
        
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        batch_size, seq_len = item_seq.shape
        mask = item_seq.gt(0)

        # Generate position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get enhanced positional embeddings
        if self.use_context_aware_pe:
            pos_emb = self.position_encoding(item_emb, position_ids)
        else:
            pos_emb = self.position_encoding(position_ids)

        # Combine item and position embeddings
        input_emb = item_emb + pos_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Get attention mask with relative position bias
        extended_attention_mask = self.get_attention_mask(item_seq)

        # Transformer encoding
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        # Explicit position injection for attention weights
        combined_input = torch.cat((output, pos_emb), dim=-1)

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


class COREtrmEnhancedPE(COREave):
    """
    CORE-trm with Enhanced Positional Encoding.
    
    Integrates:
    1. Relative positional encoding for capturing item order relationships
    2. Context-aware position embeddings (CAPE-inspired) that adapt to session content
    3. Recency bias to emphasize recent interactions
    
    This addresses the limitation that standard position encodings don't capture
    the unique patterns in recommendation sequences (recency effects, varying lengths).
    """
    def __init__(self, config, dataset):
        super(COREtrmEnhancedPE, self).__init__(config, dataset)
        self.net = TransNetEnhancedPE(config, dataset)

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)
        
        # Transformer path with enhanced positional encoding
        alpha_trm = self.net(item_seq, x_clean)
        trm_output = torch.sum(alpha_trm * x_noisy, dim=1)
        
        # Residual connection with average pooling
        alpha_ave = self.ave_net(item_seq)
        ave_output = torch.sum(alpha_ave * x_noisy, dim=1)
        
        # Combine transformer and average
        seq_output = trm_output + ave_output
        
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
