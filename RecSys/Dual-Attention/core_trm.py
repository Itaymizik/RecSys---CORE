import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.layers import TransformerEncoder

from core_ave import COREave


class TransNet(nn.Module):
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
            layer_norm_eps=self.layer_norm_eps,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.attn_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        return torch.where(extended_attention_mask, 0.0, -10000.0)

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(position_ids)

        input_emb = item_emb + pos_emb
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1]

        scorer_input = torch.cat((output, pos_emb), dim=-1)
        alpha = self.attn_scorer(scorer_input).squeeze(-1)
        alpha = alpha.masked_fill(~mask, -1e9)
        alpha = F.softmax(alpha, dim=1).unsqueeze(-1)
        return alpha

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TransNetDualAttention(nn.Module):
    """Dual-attention weighting as described in CORE-DA (global + recent intent)."""

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
            layer_norm_eps=self.layer_norm_eps,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Eq. (6): global-context scores e_i^(g) = MLP_g([f_i; f_m])
        self.global_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        # Eq. (7): recent-intent scores e_i^(r) = MLP_r([f_i; f_m])
        self.recent_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        return torch.where(extended_attention_mask, 0.0, -10000.0)

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)
        batch_size, seq_len = item_seq.size(0), item_seq.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(position_ids)

        input_emb = item_emb + pos_emb
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1]

        seq_lens = mask.long().sum(dim=1).clamp(min=1)
        last_idx = seq_lens - 1
        batch_idx = torch.arange(batch_size, device=item_seq.device)
        f_m = output[batch_idx, last_idx]
        f_m_expand = f_m.unsqueeze(1).expand(-1, seq_len, -1)

        global_input = torch.cat([output, f_m_expand], dim=-1)
        recent_input = torch.cat([output, f_m_expand], dim=-1)

        e_global = self.global_scorer(global_input).squeeze(-1)
        e_recent = self.recent_scorer(recent_input).squeeze(-1)

        e_global = e_global.masked_fill(~mask, -1e9)
        e_recent = e_recent.masked_fill(~mask, -1e9)

        alpha_global = F.softmax(e_global, dim=1).unsqueeze(-1)
        alpha_recent = F.softmax(e_recent, dim=1).unsqueeze(-1)

        return alpha_global, alpha_recent

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class COREtrm(COREave):
    def __init__(self, config, dataset):
        super(COREtrm, self).__init__(config, dataset)
        self.net = TransNet(config, dataset)

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)

        alpha = self.net(item_seq, x_clean)
        seq_output = torch.sum(alpha * x_noisy, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output


class COREtrmDualAttention(COREave):
    def __init__(self, config, dataset):
        super(COREtrmDualAttention, self).__init__(config, dataset)
        self.net = TransNetDualAttention(config, dataset)

        self.use_dual_gating = config['use_dual_gating'] if 'use_dual_gating' in config else True
        self.enable_recent_branch = config['enable_recent_branch'] if 'enable_recent_branch' in config else True
        self.enable_ave_residual = config['enable_ave_residual'] if 'enable_ave_residual' in config else True

        if self.use_dual_gating:
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.GELU(),
                nn.Linear(self.embedding_size, 2),
            )
        else:
            self.global_weight = config['global_weight'] if 'global_weight' in config else 0.6
            self.recent_weight = config['recent_weight'] if 'recent_weight' in config else 0.4

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)

        alpha_global, alpha_recent = self.net(item_seq, x_clean)

        h_global = torch.sum(alpha_global * x_noisy, dim=1)
        h_recent = torch.sum(alpha_recent * x_noisy, dim=1) if self.enable_recent_branch else h_global

        if self.use_dual_gating:
            gate_logits = self.fusion_gate(torch.cat([h_global, h_recent], dim=-1))
            gate = F.softmax(gate_logits, dim=-1)
            h_trm = gate[:, :1] * h_global + gate[:, 1:2] * h_recent
        else:
            h_trm = self.global_weight * h_global + self.recent_weight * h_recent

        if self.enable_ave_residual:
            alpha_ave = self.ave_net(item_seq)
            h_ave = torch.sum(alpha_ave * x_noisy, dim=1)
            seq_output = h_trm + h_ave
        else:
            seq_output = h_trm

        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
