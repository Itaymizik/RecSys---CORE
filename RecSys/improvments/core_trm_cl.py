import torch
import torch.nn as nn
import torch.nn.functional as F
from improvments.core_trm_modern import COREtrmModern
from improvments.augmentation import Augmentation

class COREtrmCL(COREtrmModern):
    def __init__(self, config, dataset):
        super(COREtrmCL, self).__init__(config, dataset)
        
        self.cl_weight = config['cl_weight'] if 'cl_weight' in config else 0.1
        self.temperature = config['temperature'] if 'temperature' in config else 0.07
        self.aug = Augmentation(config)
        
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # 1. Main Task Loss
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        
        loss_main = self.loss_fct(logits, pos_items)
        
        # 2. Contrastive Loss (InfoNCE)
        # Augment the batch to create two views
        aug_seq1 = self.aug.augment(item_seq, item_seq_len)
        aug_seq2 = self.aug.augment(item_seq, item_seq_len)
        
        # Get representations for augmented views
        # Note: We use the same model (self) to encode augmented views
        aug_output1 = self.forward(aug_seq1)
        aug_output2 = self.forward(aug_seq2)
        
        # Calculate InfoNCE
        loss_cl = self.calculate_cl_loss(aug_output1, aug_output2)
        
        return loss_main + self.cl_weight * loss_cl

    def calculate_cl_loss(self, view1, view2):
        """
        InfoNCE Loss:
        Minimize distance between positive pairs (view1[i], view2[i])
        Maximize distance between negative pairs (view1[i], view2[j])
        """
        batch_size = view1.shape[0]
        
        # Normalize representations
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        
        # Similarity matrix: batch_size x batch_size
        # sim[i, j] = similarity between view1[i] and view2[j]
        # We want diagonals (positives) to be high, off-diagonals (negatives) to be low
        logits = torch.matmul(view1, view2.transpose(0, 1)) / self.temperature
        
        # Labels are simply arange(batch_size) because i-th item in view1 matches i-th in view2
        labels = torch.arange(batch_size, device=view1.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
