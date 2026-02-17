import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


class COREave(SequentialRecommender):
    """
    CORE baseline encoder with optional hard-negative hybrid objective (CORE-HN style).

    When enabled, training loss follows:
      L = (1 - lambda) * L_CE + lambda * L_BPR
    where hard negatives for L_BPR are mined as top-K highest-scoring incorrect items.
    """

    def __init__(self, config, dataset):
        super(COREave, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.temperature = float(config['temperature'])

        self.sess_dropout = nn.Dropout(config['sess_dropout'])
        self.item_dropout = nn.Dropout(config['item_dropout'])

        # Hard-negative settings (paper Eq. 13-15)
        self.use_hard_negatives = bool(config['use_hard_negatives']) if 'use_hard_negatives' in config else False
        self.hard_neg_k = int(config['hard_neg_k']) if 'hard_neg_k' in config else 64
        self.hard_neg_lambda = float(config['hard_neg_lambda']) if 'hard_neg_lambda' in config else 0.2

        if not (0.0 <= self.hard_neg_lambda <= 1.0):
            raise ValueError('hard_neg_lambda must be in [0, 1].')

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def ave_net(self, item_seq):
        mask = item_seq.gt(0)
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        alpha = mask.to(torch.float) / denom
        return alpha.unsqueeze(-1)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        alpha = self.ave_net(item_seq)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def _hybrid_hn_loss(self, logits, pos_items):
        # Eq. (12): full-catalog CE
        ce_loss = self.loss_fct(logits, pos_items)

        if not (self.use_hard_negatives and self.training and self.hard_neg_lambda > 0):
            return ce_loss

        max_k = self.n_items - 2  # exclude padding id 0 and the positive item
        if max_k <= 0:
            return ce_loss

        k = max(1, min(self.hard_neg_k, max_k))

        # Eq. (13): top-K highest-scoring incorrect items under current model
        neg_logits = logits.clone()
        neg_logits[:, 0] = -1e9  # never mine padding id
        neg_logits.scatter_(1, pos_items.unsqueeze(1), -1e9)
        hard_neg_idx = torch.topk(neg_logits, k=k, dim=1).indices

        # Eq. (14): pairwise BPR-like term on mined hard negatives
        pos_logits = logits.gather(1, pos_items.unsqueeze(1))
        neg_topk_logits = logits.gather(1, hard_neg_idx)
        bpr_loss = F.softplus(neg_topk_logits - pos_logits).mean()

        # Eq. (15): hybrid objective
        return (1.0 - self.hard_neg_lambda) * ce_loss + self.hard_neg_lambda * bpr_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq)

        all_item_emb = self.item_dropout(self.item_embedding.weight)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature

        return self._hybrid_hn_loss(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        item = interaction[self.ITEM_ID]

        test_item_emb = self.item_embedding(item)
        test_item_emb = F.normalize(test_item_emb, dim=-1)

        scores = torch.sum(seq_output * test_item_emb, dim=-1) / self.temperature
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight
        test_item_emb = F.normalize(test_item_emb, dim=-1)

        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores

