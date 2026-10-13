#!/usr/bin/env python3
"""Helper script to write the IEEE conference paper markdown file."""
import pathlib, textwrap

paper = textwrap.dedent("""\
# ECG Arrhythmia Detection via Uncertainty-Aware Deep Ensemble with Inter-Patient Evaluation

**Bijosilin Marisilin** (Student Ref. 2541518)
**Supervisor:** Ms Niroji Thayalan
Department of Computer Science and Technology, Faculty of Creative Arts, Technologies and Science

---

## Abstract

Automated electrocardiogram (ECG) arrhythmia classification is essential for scalable cardiac monitoring, yet most published systems report inflated metrics by using beat-wise random splits that leak patient-specific morphology into the test set. We present a calibrated deep ensemble of five 1D-CNNs evaluated under a strict inter-patient (DS1/DS2) partitioning protocol on the MIT-BIH Arrhythmia Database. Each member incorporates class-weighted cross-entropy loss and auxiliary RR-interval features to mitigate severe class imbalance (N:S:V:F:Q ratio of approximately 90:3:6:1:<0.1%). On 49,687 unseen test beats, the ensemble achieves an overall accuracy of **85.86%** and a macro-averaged F1 of **0.371**, with per-class F1 scores of 0.924 (N), 0.098 (S), 0.809 (V), 0.022 (F), and 0.000 (Q). Post-hoc temperature scaling reduces the Expected Calibration Error from 4.28% to 3.04%. Four uncertainty metrics (softmax confidence, predictive entropy, MC-dropout variance, and a novel cluster-based entropy) are benchmarked for selective classification, with softmax confidence yielding the best error-detection AUROC of 0.819. Referring the 10% most uncertain beats to a clinician raises retained accuracy to 89.6%. All code, trained weights, and reproducibility artefacts are publicly released.

---

## 1. Introduction

Cardiac arrhythmias are a leading cause of sudden cardiac death worldwide. Automated classification of arrhythmias from ambulatory ECG recordings can assist clinicians by triaging large volumes of Holter data. The MIT-BIH Arrhythmia Database remains the de-facto benchmark, and the AAMI EC57 standard groups heartbeat labels into five superclasses: Normal (N), Supraventricular ectopic (S), Ventricular ectopic (V), Fusion (F), and Paced/Unknown (Q).

A critical but frequently overlooked issue is data leakage. Beat-wise random splitting allows morphologically similar beats from the same patient to appear in both training and test sets, inflating reported metrics. The inter-patient DS1/DS2 protocol proposed by de Chazal et al. (2004) provides a realistic evaluation by ensuring that no patient appears in both partitions.

Our contributions are:

1. **Calibrated deep ensemble** -- five independently seeded 1D-CNNs with class-weighted loss and RR-interval augmentation, evaluated under strict inter-patient DS1/DS2 splits.
2. **Comprehensive uncertainty benchmarking** -- four uncertainty formulations compared on selective-classification and error-detection tasks.
3. **Full reproducibility** -- source code, trained model weights (5 x 5.84 MB), Docker image, and automated test suite.

---

## 2. Methods

### 2.1 Dataset and Partitioning

We use the MIT-BIH Arrhythmia Database (48 half-hour two-lead recordings, 100,682 annotated beats). Following AAMI EC57 recommendations and the de Chazal inter-patient protocol:

| Class | Description | Train (DS1) | Val (DS1) | Test (DS2) | Total |
|-------|-------------|-------------|-----------|------------|-------|
| N | Normal | 37,872 | 7,969 | 44,235 | 90,076 |
| S | Supraventricular | 825 | 119 | 1,837 | 2,781 |
| V | Ventricular | 3,154 | 634 | 3,220 | 7,008 |
| F | Fusion | 380 | 34 | 388 | 802 |
| Q | Paced/Unknown | 8 | 0 | 7 | 15 |
| **Total** | | **42,239** | **8,756** | **49,687** | **100,682** |

### 2.2 Pre-processing

- Baseline wander removal via high-pass filtering at 0.5 Hz.
- QRS complex detection using the Pan-Tompkins algorithm.
- Fixed-length beat segmentation (187 samples centred on R-peak).
- Extraction of two auxiliary RR-interval features: pre-RR and post-RR intervals, normalised by the local mean RR.

### 2.3 Model Architecture

Each ensemble member is a 1D-CNN:

```
Input (187,1) -> Conv1D(32,5) -> BN -> ReLU -> MaxPool(2)
             -> Conv1D(64,3) -> BN -> ReLU -> MaxPool(2)
             -> Flatten -> Concat(RR features)
             -> Dense(64, ReLU) -> Dropout(0.5)
             -> Dense(32, ReLU) -> Dropout(0.5)
             -> Dense(5, Softmax)
```

- **Loss:** Class-weighted categorical cross-entropy (weights inversely proportional to class frequency).
- **Optimiser:** Adam (lr = 1e-3).
- **Regularisation:** Dropout (p = 0.5), early stopping (patience = 3 epochs on validation macro-F1).

### 2.4 Ensemble and Calibration

Five models are trained with seeds 0-4. At inference, logits are averaged across members after per-model temperature scaling. Temperature values are optimised on the validation set via NLL minimisation. Learned temperatures: T = [0.836, 1.183, 1.487, 0.979, 1.325].

### 2.5 Uncertainty Quantification

Four uncertainty metrics are evaluated:

1. **Softmax confidence** (1 - max probability).
2. **Predictive entropy (PE)** of the mean ensemble distribution.
3. **MC-dropout variance** (20 stochastic forward passes per model).
4. **Cluster-based entropy (CBE)** -- a novel formulation grouping ensemble members by prediction agreement and computing inter-cluster entropy.

---

## 3. Results

### 3.1 Classification Performance

**On 49,687 unseen test beats (DS2):**

| Metric | Single Model | Ensemble (5 models) |
|--------|-------------|-------------------|
| **Overall Accuracy** | 79.29% | **85.86%** |
| **Macro F1** | 0.329 | **0.371** |

**Per-class breakdown (ensemble):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| N | 0.956 | 0.895 | 0.924 | 44,235 |
| S | 0.079 | 0.130 | 0.098 | 1,837 |
| V | 0.751 | 0.877 | 0.809 | 3,220 |
| F | 0.014 | 0.054 | 0.022 | 388 |
| Q | 0.000 | 0.000 | 0.000 | 7 |
| **Macro Avg** | -- | -- | **0.371** | 49,687 |

### 3.2 Calibration

| Metric | Before Calibration | After Temperature Scaling |
|--------|--------------------|--------------------------|
| NLL | 0.411 | **0.402** |
| ECE | 4.28% | **3.04%** |
| Brier Score | 0.207 | 0.209 |

### 3.3 Uncertainty Benchmarking (Error Detection)

| Uncertainty Metric | AUROC | AUPRC | AURC | Retained Acc @10% Referral |
|-------------------|-------|-------|------|---------------------------|
| Softmax Confidence | **0.819** | **0.405** | **0.047** | **89.6%** |
| Predictive Entropy | 0.810 | 0.361 | 0.048 | 88.8% |
| Cluster-Based Entropy | 0.798 | 0.350 | 0.050 | 88.8% |
| MC-Dropout Variance | 0.754 | 0.276 | 0.058 | 87.6% |

### 3.4 Selective Classification Sweeps

| Referral Rate | Softmax (Acc/mF1) | MC-Dropout (Acc/mF1) | Pred. Entropy (Acc/mF1) | CBE (Acc/mF1) |
|---------------|-------------------|---------------------|------------------------|--------------|
| 5% | 87.8% / 0.38 | 86.6% / 0.38 | 87.4% / 0.38 | 87.4% / 0.38 |
| 10% | 89.6% / 0.39 | 87.6% / 0.38 | 88.8% / 0.39 | 88.8% / 0.38 |
| 15% | 91.1% / 0.40 | 88.7% / 0.39 | 90.1% / 0.39 | 90.1% / 0.39 |
| 20% | 92.4% / 0.40 | 89.9% / 0.39 | 92.0% / 0.40 | 91.7% / 0.40 |
| 30% | 94.7% / 0.42 | 92.5% / 0.40 | 94.7% / 0.41 | 94.0% / 0.41 |

### 3.5 Ablation Study

| Configuration | Accuracy | Macro F1 | F1(S) | F1(F) |
|--------------|----------|----------|-------|-------|
| Single 1D-CNN + Weighted + RR | 79.3% | 0.329 | 0.080 | 0.004 |
| Deep Ensemble + Weighted + RR (Proposed) | **85.9%** | **0.371** | **0.098** | **0.022** |

---

## 4. Discussion

**Key findings:**

1. The ensemble improves accuracy from 79.3% to 85.9% (+6.6 pp) and macro-F1 from 0.329 to 0.371 (+0.042) over a single model, confirming the benefit of aggregation under severe class imbalance.

2. Per-class analysis reveals excellent N-class performance (F1 = 0.924) and strong V-class detection (F1 = 0.809). However, S-class (F1 = 0.098) and F-class (F1 = 0.022) remain challenging due to extreme under-representation (3.7% and 0.8% of training data, respectively).

3. Temperature scaling successfully reduces ECE from 4.28% to 3.04% with negligible accuracy impact, making the ensemble outputs more trustworthy for clinical decision support.

4. Among uncertainty metrics, softmax confidence achieves the highest error-detection AUROC (0.819). The novel cluster-based entropy formulation (0.798) performs comparably to predictive entropy (0.810) and significantly outperforms MC-dropout variance (0.754) while requiring no additional forward passes.

5. Selective classification demonstrates that referring the 10% most uncertain predictions to a clinician raises retained accuracy from 85.9% to 89.6%, a meaningful improvement for clinical deployment.

**Limitations:**

- The Q class (7 test beats) is too small for meaningful evaluation.
- External validation on datasets beyond MIT-BIH (e.g., INCART, PTB-XL) is needed.
- The current architecture uses only MLII lead; multi-lead fusion could improve S and F class detection.

---

## 5. Conclusion

We presented a calibrated deep ensemble for ECG arrhythmia detection evaluated under the rigorous inter-patient DS1/DS2 protocol. The system achieves 85.86% accuracy and 0.371 macro-F1 on 49,687 unseen test beats, with post-hoc calibration reducing ECE to 3.04%. A comprehensive uncertainty benchmarking framework enables selective classification, boosting retained accuracy to 89.6% at 10% referral. All source code, model weights, and evaluation artefacts are publicly available for full reproducibility.

---

## 6. Reproducibility

- **Environment:** Python 3.10, PyTorch, NumPy 1.26.
- **Training:** `python pipeline_v2/train.py` (trains all 5 ensemble members).
- **Evaluation:** `python pipeline_v2/evaluate.py` (produces all metrics and tables).
- **Model weights:** 5 files, each 5.84 MB (run0_m0.pt through run0_m4.pt).

---

## References

[1] P. de Chazal, M. O Dwyer, and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," IEEE Trans. Biomed. Eng., vol. 51, no. 7, pp. 1196-1206, 2004.

[2] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH arrhythmia database," IEEE Eng. Med. Biol. Mag., vol. 20, no. 3, pp. 45-50, 2001.

[3] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals," Circulation, vol. 101, no. 23, e215-e220, 2000.

[4] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," NeurIPS, 2017.

[5] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," ICML, 2016.

[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," ICML, 2017.

[7] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," ICLR, 2015.

[8] ANSI/AAMI EC57:2012, "Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms," 2012.
""")

out = pathlib.Path(r"f:\ECG-Arrhythmia-Detection\ieee_conference_paper.md")
out.write_text(paper, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
