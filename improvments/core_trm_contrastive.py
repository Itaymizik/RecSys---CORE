import torch
import torch.nn.functional as F

from core_trm import COREtrm


class COREtrmContrastive(COREtrm):
    """CORE-trm with CL4SRec-style contrastive learning on augmented sessions.

    Note: contrastive loss is applied only when batch_size > 1.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.use_contrastive = config['use_contrastive'] if 'use_contrastive' in config else True
        self.cl_weight = config['cl_weight'] if 'cl_weight' in config else 0.1
        self.cl_dropout = config['cl_dropout'] if 'cl_dropout' in config else 0.2
        self.cl_temperature = config['cl_temperature'] if 'cl_temperature' in config else 0.2
        if not 0 <= self.cl_dropout < 1:
            raise ValueError('cl_dropout must be in [0, 1).')

    def augment_item_seq(self, item_seq):
        if self.cl_dropout == 0:
            return item_seq
        item_seq = item_seq.clone()
        mask = item_seq.gt(0)
        if not mask.any():
            return item_seq
        drop_prob = torch.rand_like(item_seq.float())
        drop_mask = (drop_prob < self.cl_dropout) & mask
        seq_lengths = mask.sum(dim=1)
        last_pos = (seq_lengths - 1).clamp(min=0)
        drop_mask[torch.arange(item_seq.size(0), device=item_seq.device), last_pos] = False
        item_seq = item_seq.masked_fill(drop_mask, 0)
        return item_seq

    def info_nce_loss(self, view_a, view_b):
        view_a = F.normalize(view_a, dim=-1)
        view_b = F.normalize(view_b, dim=-1)
        logits = torch.matmul(view_a, view_b.transpose(0, 1)) / self.cl_temperature
        labels = torch.arange(view_a.size(0), device=view_a.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_a + loss_b)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        all_item_emb = F.normalize(all_item_emb, dim=-1)

        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        ce_loss = self.loss_fct(logits, pos_items)

        if self.training and self.use_contrastive and item_seq.size(0) > 1:
            aug_a = self.augment_item_seq(item_seq)
            aug_b = self.augment_item_seq(item_seq)
            view_a = self.forward(aug_a)
            view_b = self.forward(aug_b)
            cl_loss = self.info_nce_loss(view_a, view_b)
            return ce_loss + self.cl_weight * cl_loss

        return ce_loss
