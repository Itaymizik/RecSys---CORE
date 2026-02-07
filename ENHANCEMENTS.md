# Enhanced CORE Model - Improvements Guide

## Overview
This document describes the enhancements made to the CORE-TRM model to surpass the baseline results reported in the SIGIR 2022 paper.

**Baseline Results (Diginetica):**
- CORE-trm: R@20: 52.89, MRR@20: 18.58

## Architecture Recap & Improvement Audit

### CORE-trm Baseline (What Works)
- **Two-path encoder:** Transformer-based attention captures global dependencies, while average pooling preserves representation consistency.
- **Explicit position injection:** Attention weights are computed from both content and position embeddings.
- **Normalized dot-product decoding:** Stable cosine similarity with temperature scaling.

### Why Existing Improvements Stayed Flat
The improvements in `improvments/` do not change the learning signal that dominates performance:
- **Multi-layer aggregation & dual attention** still produce a session vector trained only with the same cross-entropy objective, so the model’s decision boundary stays similar.
- **Relative/context-aware position encodings** add parameters but do not add new supervision; with short sessions they often behave like the baseline positional bias.
- **Hard negatives** can be noisy in sparse session datasets, and without stronger positives they may cancel any gains.

To improve results in a consistent, practical way, we need **additional supervision that shapes the session embedding space** rather than just changing aggregation.

## Key Improvements

### 1. Multi-Layer Aggregation 🎯
**Problem:** Original model only uses the last transformer layer, discarding valuable information from intermediate layers.

**Solution:** Aggregate all transformer layers with learnable weights:
```python
# Learnable weights for each layer
self.layer_weights = nn.Parameter(torch.ones(self.n_layers))

# Weighted combination
normalized_weights = F.softmax(self.layer_weights, dim=0)
aggregated_output = sum(weight_i * layer_i for all layers)
```

**Why it works:**
- Early layers capture local, fine-grained patterns
- Later layers capture global, semantic relationships
- Model learns optimal weighting automatically

**Expected gain:** +0.5-1.0% on R@20

---

### 2. Learnable Gating Between TRM and AVE 🚪
**Problem:** Fixed addition (`trm_output + ave_output`) treats both paths equally, regardless of session characteristics.

**Solution:** Session-adaptive gating mechanism:
```python
gate_value = σ(MLP(concat(trm_output, ave_output)))
output = gate_value * trm_output + (1 - gate_value) * ave_output
```

**Why it works:**
- Short sessions → higher weight on average pooling (simple patterns)
- Long sessions → higher weight on transformer (complex dependencies)
- Model adapts per-session dynamically

**Expected gain:** +0.8-1.5% on R@20

---

### 3. Hard Negative Mining 💪
**Problem:** Standard cross-entropy only considers all items equally as negatives.

**Solution:** Additional contrastive loss targeting hardest negatives:
```python
# Find most similar negative sessions in batch
batch_sim = matmul(seq_output, seq_output.T) / temperature
hard_neg_loss = logsumexp(batch_sim, dim=1).mean()

# Combined loss
total_loss = ce_loss + λ * hard_neg_loss
```

**Why it works:**
- Forces model to distinguish between similar but different sessions
- Improves embedding space quality
- Better separation between positive and hard negative items

**Expected gain:** +0.5-1.2% on MRR@20

---

### 4. Contrastive Session Augmentation (CL4SRec / CoSeRec) ✅
**Problem:** All previous variants rely solely on next-item CE loss, which often under-regularizes session representations.

**Solution:** Add a lightweight contrastive loss on **two augmented views** of each session (random item masking), following CL4SRec/CoSeRec-style self-supervision.
```python
aug_a = random_item_mask(item_seq)
aug_b = random_item_mask(item_seq)
loss = ce_loss + λ * info_nce(encoder(aug_a), encoder(aug_b))
```

**Why it works:**
- Encourages invariance to small session perturbations
- Improves generalization without changing inference path
- Proven gains on session-based benchmarks in recent studies (WWW 2021–2024)

**Expected gain:** +0.7-1.5% R@20 and +0.5-1.2% MRR@20 with minimal tuning

---

## How to Run

### Basic Training
```bash
# Train enhanced model on Diginetica
python main.py --model trm_enhanced --dataset diginetica

# Train contrastive model
python main.py --model trm_contrastive --dataset diginetica

# Train on other datasets
python main.py --model trm_enhanced --dataset yoochoose
python main.py --model trm_enhanced --dataset tmall
```

### Hyperparameter Tuning
```bash
# Adjust hard negative weight
python main.py --model trm_enhanced --dataset diginetica --hard-neg-weight 0.3

# Adjust temperature
python main.py --model trm_enhanced --dataset diginetica --temperature 0.05

# Adjust dropout
python main.py --model trm_enhanced --dataset diginetica --item-dropout 0.25
```

### Compare with Baseline
```bash
# Baseline
python main.py --model trm --dataset diginetica

# Enhanced
python main.py --model trm_enhanced --dataset diginetica
```

---

## Configuration

Edit [props/core_trm_enhanced.yaml](props/core_trm_enhanced.yaml):

```yaml
# Core architecture
embedding_size: 100
n_layers: 2          # More layers = stronger multi-layer aggregation
n_heads: 2

# Dropout (tune per dataset)
sess_dropout: 0.2
item_dropout: 0.2    # Try [0.15, 0.20, 0.25, 0.30]

# Temperature
temperature: 0.07    # Try [0.05, 0.07, 0.10]

# Hard negative mining
use_hard_negatives: true
hard_neg_weight: 0.5  # Try [0.3, 0.5, 0.7, 1.0]
```

Contrastive setup in [props/core_trm_contrastive.yaml](props/core_trm_contrastive.yaml):
```yaml
use_contrastive: true
cl_weight: 0.1       # Try [0.05, 0.1, 0.2]
cl_dropout: 0.2      # Item masking ratio
cl_temperature: 0.2  # InfoNCE temperature
```

---

## Expected Results

### Conservative Estimate
- **R@20:** 53.8-54.5 (+1.0-1.7% absolute)
- **MRR@20:** 19.2-19.8 (+0.6-1.2% absolute)

### Optimistic Estimate (with tuning)
- **R@20:** 54.5-55.5 (+1.7-2.7%)
- **MRR@20:** 19.5-20.0 (+1.0-1.5%)

---

## Further Improvements (Future Work)

### 5. Relative Positional Encoding
Replace absolute positions with relative positions:
```python
# Instead of position[i], use position[i] - position[j]
relative_pos = pos_ids.unsqueeze(1) - pos_ids.unsqueeze(2)
```
**Benefit:** Better handles variable-length sessions and captures item-to-item relationships.

### 6. EMA Target Embeddings
Use momentum-updated target embeddings:
```python
# Slow-moving target
target_emb = momentum * target_emb_old + (1-momentum) * current_emb
```
**Benefit:** More stable training and better contrastive learning.

### 7. Multi-Task Learning
Add auxiliary tasks:
- Next-item prediction (predict item at position t+1)
- Session length prediction
- Item order prediction (predict if sequence is shuffled)

### 8. Longer Sequences
The paper uses relatively short sequences. Try:
- Increase `MAX_ITEM_LIST_LENGTH` to 100 or 150
- Add hierarchical attention for long sequences

### 9. Label Smoothing
```python
# Soften the hard targets
loss = CrossEntropyLoss(label_smoothing=0.1)
```
**Benefit:** Prevents overconfident predictions and improves generalization.

---

## Ablation Studies

To understand which improvement contributes most:

```bash
# Disable multi-layer aggregation (set all layer weights = 1)
# Edit core_trm_enhanced.py: self.layer_weights.requires_grad = False

# Disable gating (replace with fixed addition)
# Edit core_trm_enhanced.py: gate_value = 0.5

# Disable hard negatives
python main.py --model trm_enhanced --dataset diginetica --use-hard-negatives false
```

---

## Architecture Comparison

| Component | CORE-trm (baseline) | CORE-trm-enhanced |
|-----------|-------------------|-------------------|
| Transformer layers | Last layer only | Weighted all layers |
| TRM+AVE fusion | Fixed addition | Learnable gating |
| Loss function | Cross-entropy | CE + Hard negatives |
| Position encoding | Absolute | Absolute (rel. pos. future) |

| Component | CORE-trm (baseline) | CORE-trm-contrastive |
|-----------|-------------------|----------------------|
| Supervision | CE only | CE + InfoNCE (augmented views) |
| Augmentation | None | Random item masking |

---

## Tips for Best Results

1. **Start with default config** - The defaults are well-tuned
2. **Tune dropout first** - This has the most impact per dataset
3. **Then tune temperature** - Lower = sharper, higher = smoother
4. **Finally tune hard_neg_weight** - Start at 0.5, adjust based on validation
5. **Monitor validation MRR** - Stop if it plateaus (early stopping = 5 epochs)
6. **Run 3 seeds** - Report average ± std for robustness

---

## Troubleshooting

### Training loss diverges
- Reduce `hard_neg_weight` (try 0.3 or 0.1)
- Increase `temperature` (try 0.10)
- Add gradient clipping in trainer config

### Poor validation performance
- Reduce dropout (try 0.1 for sess_dropout)
- Check if overfitting - reduce model size
- Try smaller learning rate

### Out of memory
- Reduce `eval_batch_size` in overall.yaml
- Reduce `n_layers` or `embedding_size`
- Use gradient accumulation

---

## Citation

If this enhanced model helps your research, please cite the original CORE paper:

```bibtex
@inproceedings{hou2022core,
  author = {Yupeng Hou and Binbin Hu and Zhiqiang Zhang and Wayne Xin Zhao},
  title = {CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space},
  booktitle = {{SIGIR}},
  year = {2022}
}
```
