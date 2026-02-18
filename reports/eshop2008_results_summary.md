# eshop2008 Results Summary

Date: 2026-02-18

## Default Runs (same style as other datasets)

| Model | Best Valid Recall@20 | Best Valid MRR@20 | Test Recall@20 | Test MRR@20 |
|---|---:|---:|---:|---:|
| CORE-AVE | 0.4770 | 0.1042 | 0.4789 | 0.1044 |
| CORE-TRM | 0.6758 | 0.1830 | 0.6785 | 0.1890 |
| Dual-Attention | 0.6768 | 0.1828 | 0.6769 | 0.1874 |
| Hard Negatives | 0.6770 | 0.1820 | 0.6789 | 0.1886 |
| Dual-Attention + Hard Negatives | 0.6744 | 0.1810 | 0.6784 | 0.1863 |

## Dropout Sweep (item_dropout in {0.15, 0.20, 0.25, 0.30})

Tracked CSV: `reports/eshop2008_dropout_sweep_summary.csv`

### Best by Test Recall@20

| Architecture | item_dropout | Test Recall@20 | Test MRR@20 |
|---|---:|---:|---:|
| core_trm | 0.30 | 0.6791 | 0.1883 |
| dual_attention | 0.30 | 0.6772 | 0.1863 |
| hard_negatives | 0.20 | 0.6789 | 0.1886 |
| dual_attention_hard_negatives | 0.20 | 0.6784 | 0.1863 |

### Best by Test MRR@20

| Architecture | item_dropout | Test Recall@20 | Test MRR@20 |
|---|---:|---:|---:|
| core_trm | 0.20 | 0.6785 | 0.1890 |
| dual_attention | 0.15 | 0.6761 | 0.1882 |
| hard_negatives | 0.20 | 0.6789 | 0.1886 |
| dual_attention_hard_negatives | 0.15 | 0.6781 | 0.1878 |
