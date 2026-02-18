# RecSys CORE Project — Detailed Guide

> This document is an **extended project README** for the course implementation work.
> It is intended to complement (not replace) the baseline `README.md`.

---

## 1. Project Context

This repository implements and extends the CORE session-based recommendation framework:

- **Paper**: *CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space* (SIGIR 2022)
- **Core task type**: session-based recommendation (next-item prediction)
- **Framework stack**: RecBole + PyTorch

The project itself follows a typical academic course flow:

1. pick an advanced recommender-systems topic,
2. review recent literature,
3. deeply analyze one target paper,
4. reproduce baseline results,
5. propose and implement improvements,
6. evaluate on the original and additional datasets,
7. document methodology and findings.

This structure is consistent with the project-instructions document and the repository artifacts.

---

## 2. What Exists in This Repository

### 2.1 Baseline CORE implementations

- `core_ave.py` — average-based session encoder
- `core_trm.py` — transformer-based session encoder

### 2.2 Main experiment runner

- `main.py` selects model variants, loads YAML configs, applies runtime CLI overrides, trains, validates, and evaluates.

Supported CLI model names in `main.py`:

- `ave`
- `trm`
- `trm_enhanced`
- `trm_dual_attention`
- `trm_enhanced_pe`
- `trm_hard_negatives`
- `trm_opt`
- `trm_modern`
- `trm_cl`

### 2.3 Configuration files

Under `props/`:

- `overall.yaml` — global training/evaluation settings
- `core_trm.yaml`, `core_ave.yaml` — baseline model configs
- additional variant configs (`core_trm_*.yaml`)

### 2.4 Utilities

Under `scripts/`:

- `monitor_results.py` — parses sweep outputs and appends results to CSV
- `wait_for_metrics.py` — tails logs and extracts Recall@20 / MRR@20

### 2.5 Additional project documentation

- `ENHANCEMENTS.md` — rationale and expected gains for model extensions
- `RecSys- project instructions.docx` — course workflow and report structure guidance

---

## 3. Recommended Environment

The repository provides `env_setup.sh` for reproducible setup with:

- Python 3.7
- CUDA Toolkit 10.1
- PyTorch 1.7.1
- RecBole 1.0.1

### 3.1 Quick setup

```bash
bash env_setup.sh
```

### 3.2 Manual setup (equivalent)

```bash
conda create -n core_env python=3.7 cudatoolkit=10.1 -y
conda activate core_env
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch -y
pip install recbole==1.0.1
```

---

## 4. Datasets and Evaluation Protocol

The project uses standard session datasets available in this repository structure:

- `diginetica`
- `nowplaying`
- `retailrocket`
- `tmall`
- `yoochoose`

Global evaluation defaults (from `props/overall.yaml`):

- metrics: `Recall`, `MRR`
- cutoff: `topk: [20]`
- early stopping: `stopping_step: 5`
- validation metric: `MRR@20`
- temporal random-split ratio: `RS: [0.8, 0.1, 0.1]`

---

## 5. Running Experiments

### 5.1 Baseline reproduction

```bash
python main.py --model trm --dataset diginetica
python main.py --model ave --dataset diginetica
```

### 5.2 Enhanced models

```bash
python main.py --model trm_enhanced --dataset diginetica
python main.py --model trm_dual_attention --dataset diginetica
python main.py --model trm_hard_negatives --dataset diginetica
python main.py --model trm_opt --dataset diginetica
python main.py --model trm_modern --dataset diginetica
python main.py --model trm_cl --dataset diginetica
```

### 5.3 Runtime parameter overrides

```bash
python main.py --model trm_enhanced --dataset diginetica --temperature 0.05
python main.py --model trm_enhanced --dataset diginetica --item-dropout 0.25
python main.py --model trm_hard_negatives --dataset diginetica --hard-neg-weight 0.3
python main.py --model trm_hard_negatives --dataset diginetica --use-hard-negatives true
```

### 5.4 Built-in dropout sweep

```bash
python main.py --model trm --dataset diginetica --sweep-dropout
```

The sweep evaluates `item_dropout` in `[0.15, 0.20, 0.25, 0.30]`.

---

## 6. How to Track Results

### 6.1 Monitor sweep output into CSV

```bash
python scripts/monitor_results.py path/to/train.log
```

This generates/updates:

- `core_trm_dropout_results.csv`

### 6.2 Capture Recall@20 and MRR@20 once available

```bash
python scripts/wait_for_metrics.py path/to/train.log
```

Default output:

- `core_trm_full_test_metrics.csv`

---

## 7. Improvement Directions Implemented in the Project

Based on the repository code/config naming and enhancement notes, implemented directions include:

- stronger transformer aggregation / enhanced fusion,
- dual-attention variant,
- hard negative learning,
- optimization/training refinements,
- contrastive-learning flavored variant,
- positional-encoding-enhanced version.

For details and motivation, see `ENHANCEMENTS.md` and corresponding `props/core_trm_*.yaml` files.

---

## 8. Suggested End-to-End Workflow (Course-Friendly)

A practical workflow for preparing your final report/presentation:

1. **Baseline replication** on one main dataset (e.g., Diginetica)
2. **Variant training** with default settings
3. **Ablation/tuning** (temperature, dropout, hard-negative weight)
4. **Cross-dataset validation** on at least one additional dataset
5. **Metric consolidation** with scripts into comparable tables
6. **Analysis** of what improved, what did not, and why
7. **Write-up** with clear method, setup, results, discussion, limitations, and future work

---

## 9. Expected Report Structure (Mapped to Instructions)

For the final write-up, keep this structure:

1. Abstract
2. Introduction
3. Background / Related Work
4. Method
5. Evaluation Setup
6. Results
7. Discussion
8. Conclusion (+ Future Work)

This mirrors the project-instructions document and keeps the work publication-style.

---

## 10. Known Repository Notes

- There are duplicated/parallel folders (`RecSys/`, `Dual-Attention/`, `Hard Negatives/`) from iterative experimentation.
- Prefer running from repository root with the root-level `main.py` and `props/` unless you intentionally target a submodule copy.
- Device defaults to CUDA in `props/overall.yaml`; switch to CPU if needed in constrained environments.

---

## 11. Troubleshooting

- **Import errors**: ensure the environment has matching RecBole/PyTorch versions.
- **CUDA runtime issues**: verify CUDA version compatibility or change `device` to CPU.
- **No metrics found by script**: check log formatting and ensure `Recall@20`/`MRR@20` appears in log text.
- **Unexpected score drift**: verify dataset split policy and random seed consistency.

---

## 12. Citation

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

---

## 13. Note on Source Documents

This detailed README was derived from repository-available artifacts (code, configs, enhancement notes, and course instructions docx).

`Recommendation_Systems_Article-2.pdf` is present in this environment but appears to be empty/corrupted (2 bytes), so no extractable article text was available from that file during this update.
