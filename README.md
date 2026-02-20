# RecSys CORE Project (Session-Based Recommendation)

This repository contains our project work built around **CORE** (*Simple and Effective Session-based Recommendation within Consistent Representation Space*, SIGIR 2022), including:

- Baseline reproduction of CORE models (`CORE-ave`, `CORE-trm`)
- Multiple improvement variants (enhanced fusion, dual attention, hard negatives, optimized training, contrastive learning)
- Experiment utilities for monitoring and metric extraction
- Prepared session-based datasets and RecBole configuration files

The project follows a typical academic workflow: literature grounding, baseline reproduction, method extension, and empirical evaluation on public datasets.

---

## 1) Project Goals

The main goals of this work are:

1. **Reproduce** baseline CORE performance under a unified train/valid/test protocol.
2. **Extend** the original model with practical improvements.
3. **Evaluate** improvements with standard session-based recommendation metrics (Recall@20, MRR@20).
4. **Document** the full process in a research-oriented, reproducible format.

---

## 2) Baseline CORE Architecture

CORE addresses representation inconsistency in session-based recommendation by placing session and item representations in a consistent space.

In this repository, the baseline implementation includes:

- `COREave`: average-based session encoder
- `COREtrm`: transformer-based session encoder

You can train these baseline models directly from `main.py`.

---

## 3) Extended Model Variants Included

In addition to the baseline, the project runner supports multiple variants:

- `trm_enhanced`
- `trm_dual_attention`
- `trm_enhanced_pe`
- `trm_hard_negatives`
- `trm_opt`
- `trm_modern`
- `trm_cl`

These are selectable through CLI (`--model`) and mapped in `main.py`.

---

## 4) Repository Layout (Important Files)

```text
.
├── main.py                        # Main training/evaluation entry point
├── core_trm.py                    # Baseline CORE transformer model
├── core_ave.py                    # Baseline CORE average model
├── props/
│   ├── overall.yaml               # Global training/evaluation config
│   ├── core_trm.yaml              # Baseline transformer config
│   ├── core_ave.yaml              # Baseline average config
│   └── core_trm_*.yaml            # Variant configs
├── dataset/                       # Session datasets
├── scripts/
│   ├── monitor_results.py         # Live parsing of dropout sweep logs
│   └── wait_for_metrics.py        # Wait & extract Recall@20/MRR@20
├── ENHANCEMENTS.md                # Summary of proposed project improvements
├── env_setup.sh                   # Conda-based environment setup helper
└── asset/                         # Figures used in documentation
```

> Note: The repository also contains duplicated/archival subfolders (e.g., `RecSys/`, `Dual-Attention/`, `Hard Negatives/`) that preserve intermediate project organization and experiments.

---

## 5) Environment Setup

### Option A (Recommended): use helper script

```bash
bash env_setup.sh
```

This script creates a conda environment with:

- Python 3.7
- CUDA Toolkit 10.1
- PyTorch 1.7.1
- RecBole 1.0.1

### Option B: manual setup

```bash
conda create -n core_env python=3.7 cudatoolkit=10.1 -y
conda activate core_env
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch -y
pip install recbole==1.0.1
```

---

## 6) Datasets

This project uses common session-based recommendation datasets:

- `diginetica`
- `nowplaying`
- `retailrocket`
- `tmall`
- `yoochoose`

The unified evaluation strategy uses temporal split ratios:

- Train: 80%
- Validation: 10%
- Test: 10%

(defined in `props/overall.yaml` with `eval_args.split.RS: [0.8, 0.1, 0.1]`).

---

## 7) Running Experiments

### Baseline runs

```bash
python main.py --model trm --dataset diginetica
python main.py --model ave --dataset diginetica
```

### Enhanced/variant runs

```bash
python main.py --model trm_enhanced --dataset diginetica
python main.py --model trm_dual_attention --dataset diginetica
python main.py --model trm_hard_negatives --dataset diginetica
python main.py --model trm_opt --dataset diginetica
python main.py --model trm_modern --dataset diginetica
python main.py --model trm_cl --dataset diginetica
```

### Runtime hyperparameter overrides

```bash
python main.py --model trm_enhanced --dataset diginetica --temperature 0.05
python main.py --model trm_enhanced --dataset diginetica --item-dropout 0.25
python main.py --model trm_hard_negatives --dataset diginetica --hard-neg-weight 0.3
python main.py --model trm_hard_negatives --dataset diginetica --use-hard-negatives true
```

### Built-in dropout sweep (baseline `trm`, `diginetica`)

```bash
python main.py --model trm --dataset diginetica --sweep-dropout
```

This sweep evaluates predefined `item_dropout` values `[0.15, 0.20, 0.25, 0.30]` and logs summary metrics.

---

## 8) Evaluation Settings and Metrics

Key evaluation details from `props/overall.yaml`:

- `metrics: [Recall, MRR]`
- `topk: [20]`
- `valid_metric: MRR@20`
- `stopping_step: 5`
- `eval_batch_size: 2048`

By default, experiments are configured for GPU (`device: cuda`) and 20 epochs.

---

## 9) Suggested End-to-End Research Workflow

A practical workflow for this repository:

1. **Reproduce baseline** (`trm`, `ave`) on a primary dataset (e.g., Diginetica).
2. **Confirm metric extraction** via logs and utility scripts.
3. **Run enhanced models** with default configs.
4. **Tune key parameters** (temperature, dropout, hard-negative weight).
5. **Evaluate on additional dataset(s)** for generalization.
6. **Compare and analyze** Recall@20/MRR@20 trends.
7. **Document findings** with clear tables, figures, and failure cases.

This matches the project’s academic process: baseline replication + improvement proposal + controlled evaluation.

---

## 10) Monitoring and Logging Utilities

### Live sweep result monitor

```bash
python scripts/monitor_results.py path/to/train.log
```

Creates/updates a CSV with columns:

`timestamp,rho,recall_at_20,mrr_at_20`

### Wait for final Recall@20 / MRR@20

```bash
python scripts/wait_for_metrics.py path/to/train.log
```

Writes a summary CSV (`core_trm_full_test_metrics.csv` by default).

---

## 11) Typical Baseline Reference Numbers

For Diginetica (as referenced by the original CORE README):

| Model     | Recall@20 | MRR@20 |
|-----------|-----------|--------|
| CORE-ave  | 50.21     | 18.07  |
| CORE-trm  | 52.89     | 18.58  |

Use these as directional reference points for reproduction quality before comparing enhanced variants.

---

## 12) Troubleshooting

- **CUDA mismatch / GPU unavailable**: verify CUDA and PyTorch versions, or temporarily switch to CPU in config.
- **Import/model errors**: ensure you are running from repository root with matching config/model names.
- **Unexpected metric behavior**: verify split settings and dataset preprocessing consistency.
- **Long training times**: lower epochs or batch size; run one dataset first for quick validation.

---

## 13) Citation

If you use this repository, cite CORE and RecBole:

```bibtex
@inproceedings{hou2022core,
  author = {Yupeng Hou and Binbin Hu and Zhiqiang Zhang and Wayne Xin Zhao},
  title = {CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space},
  booktitle = {SIGIR},
  year = {2022}
}

@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={CIKM},
  year={2021}
}
```
