---
title: ECG Arrhythmia AI
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app_hf.py
pinned: false
---

<div align="center">

# 🫀 ECG Arrhythmia Detection — Uncertainty-Aware Deep Ensemble

### *Calibrated 5-Member Ensemble with Inter-Patient Evaluation & Selective Classification*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PhysioNet](https://img.shields.io/badge/Dataset-MIT--BIH%20PhysioNet-0284c7?style=for-the-badge)](https://physionet.org/content/mitdb/1.0.0/)
[![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-85.86%25-10b981?style=for-the-badge)](#-results)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.371-6366f1?style=for-the-badge)](#-results)

<br/>

**Author:** [Bijosilin Marisilin](https://github.com/Bijon2002) (Student ID: `2541518`)
**Supervisor:** Ms. Niroji Thayalan
**Course:** BSc (Hons) Computer Science & Technology
**Institution:** University of Bedfordshire (UK)

---

</div>

## 📌 Project Overview

This project implements an **uncertainty-aware deep ensemble** for automated ECG arrhythmia classification, evaluated under the rigorous **AAMI EC57 inter-patient (DS1/DS2) protocol** on the MIT-BIH Arrhythmia Database. Unlike most published systems that use beat-wise random splits (which leak patient morphology into the test set), this system is tested on **entirely unseen patients**, providing realistic clinical performance estimates.

The system combines:
- A **5-member 1D-CNN deep ensemble** with class-weighted loss and RR-interval features
- **Post-hoc temperature scaling** for probability calibration
- **Four uncertainty metrics** (Softmax Confidence, Predictive Entropy, MC-Dropout Variance, Cluster-Based Entropy) for selective classification and clinician referral

---

## 📊 Results

Evaluated on **49,687 unseen test beats** (DS2 — patients never seen during training):

### Overall Performance

| Metric | Single Model | Ensemble (5 models) |
|--------|-------------|-------------------|
| **Overall Accuracy** | 79.29% | **85.86%** |
| **Macro F1-Score** | 0.329 | **0.371** |

### Per-Class Performance (Ensemble on DS2)

| Class | Description | Precision | Recall | F1-Score | Support |
|-------|-------------|-----------|--------|----------|---------|
| **N** | Normal | 0.956 | 0.895 | **0.924** | 44,235 |
| **S** | Supraventricular | 0.079 | 0.130 | **0.098** | 1,837 |
| **V** | Ventricular | 0.751 | 0.877 | **0.809** | 3,220 |
| **F** | Fusion | 0.014 | 0.054 | **0.022** | 388 |
| **Q** | Paced/Unknown | 0.000 | 0.000 | **0.000** | 7 |
| **Macro Avg** | | — | — | **0.371** | 49,687 |

> ⚠️ **Note on Accuracy:** Overall accuracy (85.86%) is dominated by the Normal class (~89% of test data). Macro-F1 (0.371) is the primary clinical metric as it treats all classes equally.

### Calibration (Temperature Scaling)

| Metric | Before Calibration | After Calibration |
|--------|--------------------|------------------|
| NLL | 0.411 | **0.402** |
| ECE | 4.28% | **3.04%** |
| Brier Score | 0.207 | 0.209 |

### Uncertainty & Selective Classification

| Uncertainty Metric | AUROC | AUPRC | Retained Acc @10% Referral |
|-------------------|-------|-------|---------------------------|
| **Softmax Confidence** | **0.819** | **0.405** | **89.6%** |
| Predictive Entropy | 0.810 | 0.361 | 88.8% |
| Cluster-Based Entropy | 0.798 | 0.350 | 88.8% |
| MC-Dropout Variance | 0.754 | 0.276 | 87.6% |

Referring the **10% most uncertain** predictions to a clinician raises retained accuracy from **85.86% → 89.6%**.

---

## 🏛️ System Architecture

```
MIT-BIH Arrhythmia Database (100,682 beats)
           │
           ├── DS1 (Training Patients)     ── Train: 42,239 | Val: 8,756 beats
           └── DS2 (Unseen Test Patients)  ── Test:  49,687 beats
                          │
              [Pre-processing Pipeline]
              • High-pass filter 0.5 Hz (baseline wander removal)
              • Pan-Tompkins QRS detection
              • 187-sample beat segmentation
              • RR-interval feature extraction (pre-RR, post-RR)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  [1D-CNN seed0]   [1D-CNN seed1]  ... [1D-CNN seed4]
  (5.84 MB each)
        └─────────────────┼─────────────────┘
                          ▼
              [Temperature Scaling Calibration]
              Temperatures: [0.836, 1.183, 1.487, 0.979, 1.325]
                          │
                          ▼
              [Ensemble Aggregation + UQ Engine]
              ├── Softmax Confidence
              ├── Predictive Entropy (PE)
              ├── MC-Dropout Variance
              └── Cluster-Based Entropy (CBE) [Novel]
                          │
                          ▼
              [Selective Classification]
              Flag uncertain beats → clinician referral
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
├── models/
│   ├── run0_m0.pt              # Ensemble member 1 (seed 0) — 5.84 MB
│   ├── run0_m1.pt              # Ensemble member 2 (seed 1) — 5.84 MB
│   ├── run0_m2.pt              # Ensemble member 3 (seed 2) — 5.84 MB
│   ├── run0_m3.pt              # Ensemble member 4 (seed 3) — 5.84 MB
│   └── run0_m4.pt              # Ensemble member 5 (seed 4) — 5.84 MB
├── results/
│   ├── weighted_rr_run0.json   # Full evaluation metrics (real numbers)
│   ├── tables.tex              # LaTeX tables for IEEE paper
│   ├── figs/                   # Figures (confusion matrix, calibration plots)
│   └── plots/                  # Additional visualisation outputs
├── data/                       # Data directory (large .npz files not tracked)
├── frontend/
│   ├── index.html              # Clinical telemetry dashboard
│   ├── app.js                  # Real-time ECG visualisation
│   └── style.css               # Medical UI design system
├── api.py                      # Flask REST API backend
├── app.py                      # Terminal inference engine
├── ieee_conference_paper.md    # IEEE paper content (this submission)
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
This produces `results/weighted_rr_run0.json` with all metrics.

### 5. Launch Clinical Dashboard
```bash
python api.py
# Open frontend/index.html in browser
```

---

## 📡 API Reference

### `POST /predict`
```json
// Request
{ "signal": [0.012, 0.045, -0.018, "...187 floats..."] }

// Response
{
  "predicted_class": "V",
  "confidence": 0.875,
  "calibrated_probabilities": { "N": 0.05, "S": 0.02, "V": 0.875, "F": 0.04, "Q": 0.015 },
  "uncertainty": {
    "softmax_confidence": 0.125,
    "predictive_entropy": 0.312,
    "mc_dropout_variance": 0.008,
    "cluster_based_entropy": 0.251
  },
  "referral_flag": false
}
```

### `GET /random_beat`
Returns a random annotated beat from the MIT-BIH test set.

---

## 🔬 Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet)
- 48 half-hour two-lead ECG recordings
- 100,682 annotated beats across 5 AAMI classes
- Inter-patient split: DS1 (training, records 101-124) / DS2 (test, records 200-234)

| Class | Description | Total Beats | % |
|-------|-------------|-------------|---|
| N | Normal | 90,076 | 89.5% |
| S | Supraventricular | 2,781 | 2.8% |
| V | Ventricular | 7,008 | 7.0% |
| F | Fusion | 802 | 0.8% |
| Q | Paced/Unknown | 15 | <0.1% |

---

## 📄 IEEE Conference Paper

The full paper content (ready for LaTeX typesetting) is in [`ieee_conference_paper.md`](ieee_conference_paper.md).

```bibtex
@inproceedings{marisilin2026ecg,
  author    = {Bijosilin Marisilin},
  title     = {ECG Arrhythmia Detection via Uncertainty-Aware Deep Ensemble
               with Inter-Patient Evaluation},
  booktitle = {IEEE International Conference on Biomedical Engineering},
  year      = {2026},
  note      = {Supervisor: Ms. Niroji Thayalan, University of Bedfordshire}
}
```

---

## 🎓 Academic Context

This project was developed as a **Final Year Undergraduate Research Project (CIS017-3)** at the **University of Bedfordshire (UK)**.

**Supervisor:** Ms. Niroji Thayalan
**Student:** Bijosilin Marisilin (ID: 2541518)

---

## 📄 License

Released under the [MIT License](LICENSE). For academic, research, and non-commercial use only.
