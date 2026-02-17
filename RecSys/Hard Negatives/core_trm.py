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

        self.fn = nn.Sequential(
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


class COREtrm(COREave):
    """Original CORE-trm architecture with optional hard-negative loss inherited from COREave."""

    def __init__(self, config, dataset):
        super(COREtrm, self).__init__(config, dataset)
        self.net = TransNet(config, dataset)

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)

        alpha_trm = self.net(item_seq, x_clean)
        trm_output = torch.sum(alpha_trm * x_noisy, dim=1)

        # Keep the residual average path used in your current CORE-trm file.
        alpha_ave = self.ave_net(item_seq)
        ave_output = torch.sum(alpha_ave * x_noisy, dim=1)

        seq_output = trm_output + ave_output
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
