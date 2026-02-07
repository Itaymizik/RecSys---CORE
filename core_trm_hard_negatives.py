import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder

from core_ave import COREave


class TransNetWithHardNeg(nn.Module):
    """
    Transformer Network designed for hard negative sampling training.
    Same architecture as baseline but optimized for better training dynamics.
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

        # Attention aggregation with position awareness
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

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        # Explicit position injection
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


class COREtrmHardNeg(COREave):
    """
    CORE-trm with Optimized Negative Sampling Training Strategy.
    
    Key improvements:
    1. Hard negative mining: samples top-k predicted items as negatives
    2. Hybrid loss: combines cross-entropy with BPR pairwise ranking loss
    3. Adaptive negative sampling: focuses on confusing items
    
    Inspired by TRON (RecSys 2023) which achieved 18% CTR improvement
    through optimized negative sampling and listwise ranking loss.
    """
    def __init__(self, config, dataset):
        super(COREtrmHardNeg, self).__init__(config, dataset)
        self.net = TransNetWithHardNeg(config, dataset)
        
        # Hard negative sampling parameters
        self.use_hard_negatives = config['use_hard_negatives'] if 'use_hard_negatives' in config else True
        self.num_hard_negatives = config['num_hard_negatives'] if 'num_hard_negatives' in config else 128
        self.hard_neg_weight = config['hard_neg_weight'] if 'hard_neg_weight' in config else 0.3
        
        # Loss function configuration
        self.loss_type = config['loss_type']
        self.use_bpr_loss = config['use_bpr_loss'] if 'use_bpr_loss' in config else True
        self.bpr_weight = config['bpr_weight'] if 'bpr_weight' in config else 0.2
        
        # Margin for ranking losses
        self.margin = config['ranking_margin'] if 'ranking_margin' in config else 0.5

    def forward(self, item_seq):
        x_clean = self.item_embedding(item_seq)
        x_noisy = self.sess_dropout(x_clean)
        
        # Transformer path
        alpha_trm = self.net(item_seq, x_clean)
        trm_output = torch.sum(alpha_trm * x_noisy, dim=1)
        
        # Residual connection with average pooling
        alpha_ave = self.ave_net(item_seq)
        ave_output = torch.sum(alpha_ave * x_noisy, dim=1)
        
        # Combine
        seq_output = trm_output + ave_output
        
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def sample_hard_negatives(self, seq_output, pos_items, all_item_emb):
        """
        Sample hard negatives based on current model predictions.
        Selects items with highest scores (excluding positives) as challenging negatives.
        
        Args:
            seq_output: [batch_size, hidden_size] - session representations
            pos_items: [batch_size] - ground truth next items
            all_item_emb: [n_items, hidden_size] - all item embeddings
            
        Returns:
            hard_neg_items: [batch_size, num_hard_negatives] - sampled hard negative item IDs
        """
        batch_size = seq_output.size(0)
        
        # Compute scores for all items
        with torch.no_grad():
            all_scores = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
            # [batch_size, n_items]
            
            # Mask out positive items (set to very low score)
            all_scores[torch.arange(batch_size), pos_items] = -1e9
            
            # Select top-k items as hard negatives
            # These are items the model currently predicts as likely, but are wrong
            _, hard_neg_items = torch.topk(all_scores, k=self.num_hard_negatives, dim=1)
            # [batch_size, num_hard_negatives]
        
        return hard_neg_items

    def calculate_bpr_loss(self, seq_output, pos_item_emb, neg_item_emb):
        """
        Bayesian Personalized Ranking (BPR) pairwise loss.
        Encourages positive items to rank higher than negative items.
        
        Args:
            seq_output: [batch_size, hidden_size]
            pos_item_emb: [batch_size, hidden_size]
            neg_item_emb: [batch_size, num_negatives, hidden_size]
            
        Returns:
            bpr_loss: scalar
        """
        # Positive scores: [batch_size, 1]
        pos_scores = (seq_output * pos_item_emb).sum(dim=-1, keepdim=True) / self.temperature
        
        # Negative scores: [batch_size, num_negatives]
        neg_scores = torch.matmul(seq_output.unsqueeze(1), neg_item_emb.transpose(1, 2)).squeeze(1) / self.temperature
        
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        # Equivalent to: log(1 + exp(neg_score - pos_score))
        score_diff = neg_scores - pos_scores  # [batch_size, num_negatives]
        bpr_loss = F.softplus(score_diff).mean()
        
        return bpr_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        
        # === Standard Cross-Entropy Loss ===
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        ce_loss = self.loss_fct(logits, pos_items)
        
        total_loss = ce_loss
        
        # === Hard Negative Mining + BPR Loss ===
        if self.use_hard_negatives and self.training:
            # Sample hard negatives
            hard_neg_items = self.sample_hard_negatives(seq_output, pos_items, all_item_emb)
            
            # Get embeddings
            pos_item_emb = all_item_emb[pos_items]  # [batch_size, hidden_size]
            hard_neg_emb = all_item_emb[hard_neg_items]  # [batch_size, num_hard_neg, hidden_size]
            
            if self.use_bpr_loss:
                # Add BPR pairwise ranking loss
                bpr_loss = self.calculate_bpr_loss(seq_output, pos_item_emb, hard_neg_emb)
                total_loss = (1 - self.bpr_weight) * ce_loss + self.bpr_weight * bpr_loss
            else:
                # Alternative: add hard negatives to cross-entropy with higher weight
                # Compute scores for hard negatives
                hard_neg_scores = torch.matmul(
                    seq_output.unsqueeze(1), 
                    hard_neg_emb.transpose(1, 2)
                ).squeeze(1) / self.temperature
                
                # Create weighted CE: penalize high scores on hard negatives
                hard_neg_loss = F.softplus(hard_neg_scores).mean()
                total_loss = ce_loss + self.hard_neg_weight * hard_neg_loss
        
        return total_loss
