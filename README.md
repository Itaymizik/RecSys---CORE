# Enhanced CORE Framework: Dual-Attention and Hard-Negative Extensions

This is the extended and enhanced PyTorch implementation based on the original [CORE paper](https://arxiv.org/abs/2204.11067) for Session-based Recommendation within Consistent Representation Space.

**Project Authors:** Liran Smadja, Itay Mizikov (Ben-Gurion University, Beer-Sheva, Israel)
**Repository:** [RecSys---CORE](https://github.com/Itaymizik/RecSys---CORE)

> Yupeng Hou, Binbin Hu, Zhiqiang Zhang, Wayne Xin Zhao. CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space. SIGIR 2022 short.

## Overview

We argue that session embedding encoded by non-linear encoder is usually not in the same representation space as item embeddings, resulting in the inconsistent prediction issue. In this work, we aim at unifying the representation space throughout the encoding and decoding process in session-based recommendation, and propose a simple and effective framework named CORE.

<div  align="center"> 
<img src='asset/abs.png' width="50%">
</div>

## 🧩 Methodology & Architecture (CORE-DA / CORE-HN / CORE-DAHN)

The following figures summarize (A) the end-to-end experimental pipeline we use and (B) the CORE-DAHN encoder + hybrid objective that combines dual-attention with hard-negative learning.

> **Note:** Make sure the following files exist in your repo under `asset/`:
> - `asset/DA_HN_Pipeline.png`
> - `asset/DA_HN_Diagram.png`

<div align="center">
  <img src="DA_HN_Pipeline.png" width="95%">
  <p><em>Figure 1: Overall experimental pipeline used in CORE-DAHN, from session construction and embedding lookup, through dual-attention + gated fusion, full-catalog scoring, hard-negative mining, optimization, and evaluation.</em></p>
</div>

<div align="center">
  <img src="DA_HN_Diagram.png" width="95%">
  <p><em>Figure 2: CORE-DAHN details. (A) Dual-Attention Encoder: global-context and recent-intent attentions fused via a learnable gate + residual mean pooling. (B) Hybrid objective: Cross-Entropy with the positive item plus a BPR-style term over mined hard negatives.</em></p>
</div>

## ✨ New Features & Enhancements

This repository extends the baseline CORE models (`trm` and `ave`) with architectural improvements focused on ranking-quality gains while maintaining [representation consistency](https://arxiv.org/abs/2204.11067):

* **CORE-DA (Dual Attention Mechanism)**: Enhances representation learning by computing two sets of importance weights: one capturing the global session context and another explicitly emphasizing recent intent via the last-item query. These are fused via an adaptive learnable gate.
* **CORE-HN (Hard Negative Mining)**: Introduces a contrastive learning approach that explicitly mines and penalizes hard negative samples (Top-K highest scoring incorrect items) to improve the quality of the embedding space. This uses a BPR-style pairwise objective added to standard cross-entropy.
* **CORE-DAHN (Combined Approach)**: A unified model combining both Dual Attention and Hard Negative Mining for targeted gains on specific benchmarks while preserving representation consistency.

## 📁 Repository Structure

The codebase is organized into different modules for clarity and comparison:
*   `Dual-Attention/`: Implementation of the Dual Attention (`trm_da`) approach.
*   `Hard Negatives/`: Implementation of the Hard Negatives mining approach on top of CORE.
*   `Dual-Attention-Hard-Negatives/`: Implementation of the combined approach.
*   `results/`: Directory containing extensive training logs and evaluation summaries, including dropout sweeps and experiment results.
*   `dataset/`: Subdirectory for dataset loading and preprocessing.

## Requirements

```
recbole==1.0.1
python==3.7
pytorch==1.7.1
cudatoolkit==10.1
```

## Datasets

You can download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1dlJ3PzcT5SCN8-Mocr_AIQPGk9DVgTWB?usp=sharing). Then,
```bash
mkdir -p dataset
mv DATASET.zip dataset
cd dataset
unzip DATASET.zip
cd ..
```

Supported `DATASET` options:
*   `diginetica`
*   `nowplaying`
*   `retailrocket`
*   `tmall`
*   `yoochoose`
*   **`eshop2008`** (Newly integrated format)

## 🚀 Usage & Reproduction

To evaluate and train the models, navigate to the respective directory and use the `main.py` script. 

### 1. Baseline CORE models
Run the original `trm` or `ave` models from the root directory:
```bash
# Example: Train baseline CORE-trm
python main.py --model trm --dataset diginetica
```

### 2. Dual Attention
Navigate to the `Dual-Attention` directory:
```bash
cd Dual-Attention
python main.py --model trm_da --dataset diginetica
```

### 3. Hard Negatives
Navigate to the `Hard Negatives` directory:
```bash
cd "Hard Negatives"
python main.py --model trm --dataset diginetica --use-hard-negatives True
```

### 4. Combined (Dual Attention + Hard Negatives)
Navigate to the `Dual-Attention-Hard-Negatives` directory:
```bash
cd Dual-Attention-Hard-Negatives
python main.py --model trm_da --dataset diginetica --use-hard-negatives True
```

**Custom Hyperparameters:**
You can also override defaults via command-line arguments:
*   `--temperature 0.07`
*   `--item-dropout 0.2`
*   Root `main.py`: `--hard-neg-weight 0.5`
*   `Hard Negatives/` and `Dual-Attention-Hard-Negatives/`: `--hard-neg-lambda 0.5`

## 📊 Results

The table below reports single-run test results (with seed = 2020) on the five CORE benchmarks, plus an additional evaluation on `eShop2008`. Metrics are expressed as **Recall@20 / MRR@20 (%)**.

| Dataset | CORE-ave | CORE-trm | CORE-DA | CORE-HN | CORE-DAHN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diginetica** | 50.28 / 17.99 | 52.87 / 18.66 | 52.90 / 18.66 | 52.52 / 18.46 | 52.67 / **18.69** |
| **Nowplaying** | 20.24 / 6.64 | 21.73 / 7.45 | 21.77 / **7.67** | 21.70 / 7.43 | 21.46 / 7.64 |
| **RetailRocket** | 58.58 / 37.18 | 61.80 / 38.59 | 61.95 / 38.80 | 61.82 / 38.69 | 61.83 / **38.85** |
| **Tmall** | 45.02 / 31.62 | 44.99 / 31.63 | 45.01 / 31.63 | 44.92 / **31.76** | 44.92 / 31.75 |
| **Yoochoose** | 59.02 / 25.07 | 64.58 / 28.21 | 64.60 / 28.29 | 64.56 / **28.31** | 64.43 / 28.19 |
| **eShop2008** | – | 67.84 / **18.89** | 67.61 / 18.82 | 67.83 / 18.82 | 67.81 / 18.78 |

**Notes:** 
* Dual Attention yields small but consistent rank-quality gains (MRR@20) over CORE-trm on most datasets. 
* Hard Negatives provide dataset-dependent benefits (e.g., Tmall and Yoochoose). 
* The Combined approach achieved the highest MRR on Diginetica and RetailRocket.
* On eShop2008, CORE-trm remains the strongest baseline in this comparison.

<img src="asset/res.png" width="50%">

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole) and [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN).

Please cite the following papers as the references if you use our codes or the processed datasets.

```
@inproceedings{hou2022core,
  author = {Yupeng Hou and Binbin Hu and Zhiqiang Zhang and Wayne Xin Zhao},
  title = {CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space},
  booktitle = {{SIGIR}},
  year = {2022}
}


@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
```
