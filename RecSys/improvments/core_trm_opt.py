import torch
import torch.nn as nn
import torch.nn.functional as F
from core_trm import COREtrm

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

class COREtrmOpt(COREtrm):
    def __init__(self, config, dataset):
        super(COREtrmOpt, self).__init__(config, dataset)
        
        # Load hyperparams
        self.label_smoothing = config['label_smoothing'] if 'label_smoothing' in config else 0.0
        
        if self.label_smoothing > 0:
            self.loss_fct = LabelSmoothingLoss(smoothing=self.label_smoothing)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        
        loss = self.loss_fct(logits, pos_items)
        return loss
