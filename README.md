<div align="center">

# 🫀 ECG Arrhythmia Detection — Uncertainty-Aware Deep Ensemble

### *Calibrated 5-Member Ensemble with Inter-Patient Evaluation & Selective Classification*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PhysioNet](https://img.shields.io/badge/Dataset-MIT--BIH%20PhysioNet-0284c7?style=flat-square)](https://physionet.org/content/mitdb/1.0.0/)
[![Ensemble Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-85.86%25-10b981?style=flat-square)](#-results)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.371-6366f1?style=flat-square)](#-results)
[![Calibration](https://img.shields.io/badge/ECE-3.04%25-blue?style=flat-square)](#calibration-temperature-scaling)

<br/>

**Author:** [Bijosilin Marisilin](https://github.com/Bijon2002) (Student ID: `2541518`)  
**Supervisor:** Ms. Niroji Thayalan  
**Course:** BSc (Hons) Computer Science & Technology  
**Institution:** University of Bedfordshire (UK) / SLIIT Northern Uni 

---

</div>

## 📌 Project Overview

This project implements an **uncertainty-aware deep ensemble** for automated ECG arrhythmia classification, evaluated under the rigorous **AAMI EC57 inter-patient (DS1/DS2) protocol** on the MIT-BIH Arrhythmia Database. Unlike most published systems that use beat-wise random splits (which leak patient morphology into the test set), this system is tested on **entirely unseen patients**, providing realistic clinical performance estimates.

The system combines:
- A **5-member 1D-CNN deep ensemble** with class-weighted loss and RR-interval timing features
- **Post-hoc temperature scaling** for probability calibration
- **Four uncertainty metrics** (Softmax Confidence, Predictive Entropy, MC-Dropout Variance, Cluster-Based Entropy) for selective classification and clinician referral

---

## 📊 Results

Evaluated on **49,687 unseen test beats** (DS2 — 22 patients never seen during training):

### Overall Performance

| Metric | Single 1D-CNN Baseline | Deep Ensemble (5 models) |
|:---|:---:|:---:|
| **Overall Accuracy** | 79.29% | **85.86%** |
| **Macro F1-Score (5 classes)** | 0.329 | **0.371** |
| **Macro F1-Score (N, S, V, F)** | 0.411 | **0.463** |

### Per-Class Performance (Ensemble on DS2)

| Class | Description | Precision | Recall (Sens.) | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
| **N** | Normal | 0.956 | 0.895 | **0.924** | 44,235 |
| **S** | Supraventricular | 0.079 | 0.130 | **0.098** | 1,837 |
| **V** | Ventricular | 0.751 | 0.877 | **0.809** | 3,220 |
| **F** | Fusion | 0.014 | 0.054 | **0.022** | 388 |
| **Q** | Paced / Unknown | 0.000 | 0.000 | **0.000** | 7 |
| **Macro Avg** | | — | — | **0.371** | **49,687** |

> ⚠️ **Note on Accuracy:** Overall accuracy (85.86%) is dominated by the Normal class (~89% of test data). Macro-F1 (0.371) is the primary clinical metric as it treats all classes equally.

### Calibration (Temperature Scaling)

| Metric | Before Calibration (Raw) | After Temperature Scaling |
|:---|:---:|:---:|
| **Negative Log-Likelihood (NLL)** | 0.4113 | **0.4020** |
| **Expected Calibration Error (ECE)** | 4.28% | **3.04%** |
| **Brier Score** | **0.2071** | 0.2091 |

### Uncertainty & Selective Classification (Error Detection)

| Uncertainty Metric | AUROC | AUPRC | AURC | Retained Acc @ 10% Referral | Retained Acc @ 30% Referral |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Softmax Confidence** | **0.819** | **0.405** | **0.0465** | **89.6%** | **94.7%** |
| **Predictive Entropy** | 0.810 | 0.361 | 0.0480 | 88.8% | 94.7% |
| **Cluster-Based Entropy (CBE)** | 0.798 | 0.350 | 0.0495 | 88.8% | 94.0% |
| **MC-Dropout Variance** | 0.754 | 0.276 | 0.0576 | 87.6% | 92.5% |

Referring the **10% most uncertain** predictions to a clinician raises retained diagnostic accuracy from **85.86% → 89.6%** (and up to **94.7%** at 30% referral).

---

## 🏛️ System Architecture

```text
MIT-BIH Arrhythmia Database (100,682 beats)
           │
           ├── DS1 (Training Cohort)       ── Train: 42,239 | Val: 8,756 beats
           └── DS2 (Unseen Test Cohort)    ── Test:  49,687 beats
                          │
              [Pre-processing Pipeline]
              • Bandpass filter 0.5–45 Hz (baseline wander removal)
              • R-peak centered 360-sample beat segmentation (1.0 s)
              • 4 RR-interval timing features (pre-RR, post-RR, local context, ratio)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  [1D-CNN seed0]   [1D-CNN seed1]  ... [1D-CNN seed4]
        └─────────────────┼─────────────────┘
                          ▼
              [Temperature Scaling Calibration]
              Learned T: [0.836, 1.183, 1.487, 0.979, 1.325]
                          │
                          ▼
              [Ensemble Averaging & Uncertainty Engine]
              ├── Softmax Confidence (SC)
              ├── Predictive Entropy (PE)
              ├── Cluster-Based Entropy (CBE) [Clinical Risk Tiers]
              └── MC-Dropout Epistemic Variance
                          │
                          ▼
              [Selective Classification Triage]
              Flag uncertain beats → Human cardiologist review
```

---

## 🗂️ Project Structure

```text
ECG-Arrhythmia-Detection/
├── pipeline_v2/
│   ├── prepare_data.py         # Inter-patient DS1/DS2 data partitioning
│   ├── train.py                # 5-member ensemble training (class-weighted)
│   ├── evaluate.py             # Full evaluation: metrics, calibration, UQ
│   ├── common.py               # Shared model architecture & utilities
│   └── metrics.py              # F1, ECE, AUROC, AURC computation
├── models/                     # Trained PyTorch model checkpoints (.pt)
├── results/
│   ├── weighted_rr_run0.json   # Full evaluation metrics (exact numbers)
│   ├── tables.tex              # Ready LaTeX tables for IEEE paper
│   └── figs/                   # Figures (confusion matrix, referral curves)
├── data/                       # Data directory (large .npz files excluded via .gitignore)
├── frontend/                   # Clinical telemetry dashboard (HTML/JS/CSS)
├── api.py                      # Flask REST API backend
├── app.py                      # Inference engine
├── main.tex                    # IEEE Conference paper LaTeX source code
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/Bijon2002/ECG-Arrhythmia-Detection.git
cd ECG-Arrhythmia-Detection

conda create -n ecg_env python=3.10
conda activate ecg_env
pip install -r requirements.txt
```

### 2. Prepare Data (Inter-Patient DS1/DS2 Split)
```bash
python pipeline_v2/prepare_data.py
```

### 3. Train All 5 Ensemble Members
```bash
python pipeline_v2/train.py
```

### 4. Run Full Evaluation
```bash
python pipeline_v2/evaluate.py
```
This evaluates all 49,687 unseen beats, fits temperature scaling, benchmarks uncertainty metrics, and saves `results/weighted_rr_run0.json`.

---

## 📄 IEEE Conference Paper

The paper source code is available in [`main.tex`](main.tex).

```bibtex
@inproceedings{marisilin2026ecg,
  author    = {Bijosilin Marisilin and Niroji Thayalan},
  title     = {Uncertainty-Aware ECG Arrhythmia Classification Using Deep Ensembles with Inter-Patient Evaluation},
  booktitle = {IEEE Conference on Biomedical Engineering and Health Informatics},
  year      = {2026}
}
```

---

## 🎓 Academic Context

Developed as an Undergraduate Research Project at the **University of Bedfordshire (UK)** in collaboration with **SLIIT Northern University**.

* **Author:** Bijosilin Marisilin (Student ID: `2541518`)
* **Supervisor:** Ms. Niroji Thayalan

---

## 📄 License

Released under the [MIT License](LICENSE). For academic, research, and educational use.
