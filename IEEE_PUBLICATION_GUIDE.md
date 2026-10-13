# IEEE Publication Blueprint & Execution Guide

**Project:** Clinically-Calibrated Deep Ensemble with Risk-Stratified Selective Rejection for Inter-Patient ECG Arrhythmia Classification  
**Target Venues:** IEEE EMBC, IEEE BHI, IEEE CBMS, IEEE MeMeA, or IEEE Access  
**Repository Package:** `pipeline_v2/`

---

## 📌 1. The Core Scientific Strategy

### Why the Original Thesis Results Needed Transformation for IEEE:
1. **The Data Leakage Problem:**
   * Standard 80/20 beat-wise splitting mixes beats from the same subject into both training and testing sets (**Intra-patient split**).
   * This yields inflated benchmark figures (~98% accuracy), which biomedical IEEE reviewers treat as a fatal flaw because the model memorizes individual patient wave signatures.
2. **The "Two-Split" Contribution:**
   * Instead of hiding the 98% result, we report **both paradigms side-by-side**:
     * **Beat-Wise Split:** Shows the ~98% benchmark with data leakage.
     * **Inter-Patient Split (de Chazal DS1/DS2):** Shows realistic performance on 22 completely unseen patients (~81%–86%).
   * Quantifying this gap is itself a recognized scientific contribution that proves methodological honesty.
3. **Rescuing the Supraventricular ($S$) Class with RR Features:**
   * On unseen patients, $S$ beats share almost identical QRS morphology with Normal ($N$) beats.
   * By fusing **4 RR-interval timing features** (pre-RR, post-RR, local context RR, and prematurity ratio), the model learns temporal rhythm context, drastically improving $S$ detection.
4. **Selective Classification (Abstention) as the Main Novelty:**
   * Rather than forcing the AI to guess on ambiguous beats, we use **Cluster-Based Entropy (CBE)** and calibrated uncertainty to sweep referral rates ($0\% \to 5\% \to 10\% \to 15\% \to 20\% \to 30\%$).
   * As ambiguous beats are routed to human cardiologists, the retained diagnostic performance rises to clinical reliability (>95%).

---

## 🗂️ 2. Pipeline Architecture (`pipeline_v2/`)

* [`pipeline_v2/common.py`](pipeline_v2/common.py): Model definition (`ECGNet` with optional RR feature concatenation), patient partitioning constants (DS1, DS2, PACED, VAL), and device utilities.
* [`pipeline_v2/prepare_data.py`](pipeline_v2/prepare_data.py): Feature extraction (360-sample bandpass waveform + 4 RR features) and dataset generation for both inter-patient and beat-wise splits.
* [`pipeline_v2/metrics.py`](pipeline_v2/metrics.py): ECE, Brier score, NLL calibration, Cluster-Based Entropy (CBE), AUROC, AUPRC, AURC, and referral sweep metrics.
* [`pipeline_v2/train.py`](pipeline_v2/train.py): 5-member Deep Ensemble trainer with random seeds, class weighting, and validation Macro F1 early stopping.
* [`pipeline_v2/evaluate.py`](pipeline_v2/evaluate.py): Temperature scaling calibration, uncertainty quantification, referral sweeps, and IEEE column-width figures.
* [`pipeline_v2/aggregate.py`](pipeline_v2/aggregate.py): Auto-compiler that parses JSON run files and writes publication-ready LaTeX tables directly to `results/tables.tex`.

---

## ⚡ 3. Terminal Execution Commands

Run all commands from the repository root: `f:\ECG-Arrhythmia-Detection`.

### Step 1: Data Preparation *(Already Completed ✅)*
```powershell
# Inter-patient DS1/DS2 split (44 MIT-BIH records, 4 RR features)
python pipeline_v2/prepare_data.py --mode interpatient --no-paced

# Beat-wise random split (for data leakage comparison table)
python pipeline_v2/prepare_data.py --mode beatwise --no-paced
```
*Outputs generated:* `data/split.npz`, `data/split_counts.json`, `data/split_beatwise.npz`, `data/split_counts_beatwise.json`.

---

### Step 2: Training the Ensembles

#### 2A. Proposed Main Model (Inter-Patient, Weighted, 4 RR Features):
```powershell
python pipeline_v2/train.py --run 0 --weighted 1 --rr 1 --split interpatient --members 5 --epochs 25 --patience 5 --bs 128
```
*Saves 5 model checkpoints to:* `models/weighted_rr/run0_m0.pt` through `run0_m4.pt`.

#### 2B. Ablation 1 (Inter-Patient without RR Features):
*Proves why RR timing features are mathematically necessary.*
```powershell
python pipeline_v2/train.py --run 0 --weighted 1 --rr 0 --split interpatient --members 5 --epochs 25 --patience 5 --bs 128
```
*Saves 5 model checkpoints to:* `models/weighted/run0_m0.pt` through `run0_m4.pt`.

#### 2C. Ablation 2 (Inter-Patient Unweighted):
*Proves why class weighting is necessary to prevent majority-class collapse.*
```powershell
python pipeline_v2/train.py --run 0 --weighted 0 --rr 0 --split interpatient --members 5 --epochs 25 --patience 5 --bs 128
```
*Saves 5 model checkpoints to:* `models/unweighted/run0_m0.pt` through `run0_m4.pt`.

#### 2D. Leakage Baseline (Beat-Wise Split with RR Features):
*Generates the ~98% baseline to prove the leakage gap in Table IV.*
```powershell
python pipeline_v2/train.py --run 0 --weighted 1 --rr 1 --split beatwise --members 5 --epochs 25 --patience 5 --bs 128
```
*Saves 5 model checkpoints to:* `models/weighted_rr_bw/run0_m0.pt` through `run0_m4.pt`.

---

### Step 3: Evaluation, Calibration & Figure Generation

#### 3A. Evaluate Main Proposed Model:
```powershell
python pipeline_v2/evaluate.py --run 0 --weighted 1 --rr 1 --split interpatient --mc 10
```
*Generates:*
* `results/weighted_rr_run0.json` (Full metrics, ECE, AUROC, sweeps)
* `results/weighted_rr_run0_scores.npz` (Calibrated probabilities and uncertainty vectors)
* `results/figs/weighted_rr_run0_f1_vs_referral.png` (IEEE Figure: Macro F1 vs. Referral Rate)
* `results/figs/weighted_rr_run0_confusion_matrix.png` (IEEE Figure: Normalized Confusion Matrix)

#### 3B. Evaluate Ablation Models:
```powershell
# Ablation 1 (Without RR)
python pipeline_v2/evaluate.py --run 0 --weighted 1 --rr 0 --split interpatient --mc 10

# Ablation 2 (Unweighted)
python pipeline_v2/evaluate.py --run 0 --weighted 0 --rr 0 --split interpatient --mc 10

# Leakage Baseline (Beat-wise)
python pipeline_v2/evaluate.py --run 0 --weighted 1 --rr 1 --split beatwise --mc 10
```

---

### Step 4: Auto-Generate LaTeX Publication Tables

Once evaluations are completed, run:
```powershell
python pipeline_v2/aggregate.py --main weighted_rr
```

*Output file generated:* [`results/tables.tex`](results/tables.tex)  
Contains formatted LaTeX code for:
1. `Table I`: Heartbeat Class Distribution across DS1/DS2 partitions.
2. `Table II`: Per-Class Diagnostic Performance on Unseen Patients (Precision, Recall, F1, Support).
3. `Table III`: Architectural Ablation Study (Single CNN vs. Ensemble, Impact of Weights and RR).
4. `Table IV`: Data Leakage Demonstration (Beat-wise Contaminated vs. Inter-patient Realistic).
5. `Table V`: Probability Calibration (NLL, ECE, Brier score before and after Temperature Scaling).
6. `Table VI`: Uncertainty Error Detection Performance (AUROC, AUPRC, AURC, Retained Acc @ 10% referral).
7. `Table VII`: Selective Classification Sweeps (Accuracy and Macro F1 on Kept Beats at 5%, 10%, 15%, 20%, 30% Referral).

---

## 📝 4. IEEE Manuscript Structure (4 to 6 Pages)

```
IEEE Two-Column Template
├── Title: "Clinically-Calibrated Deep Ensemble with Risk-Stratified Selective 
│          Rejection for Inter-Patient ECG Arrhythmia Classification"
├── Abstract (150-200 words):
│   - Context: Softmax overconfidence & inter-patient generalization drop.
│   - Method: 5-member 1D-CNN ensemble + 4 RR timing features + Temperature Scaling + CBE.
│   - Results: Inter-patient vs. beat-wise comparison; ECE reduction; selective classification gains.
├── I. Introduction:
│   - Clinical dilemma: Diagnostic uncertainty vs. alarm fatigue.
│   - The flaw of intra-patient benchmark inflation in literature.
├── II. Related Work:
│   - Deep learning for ECG (1D-CNNs, RR timing).
│   - Uncertainty quantification (MC Dropout, ensembles, temperature scaling).
├── III. Proposed Methodology:
│   - Preprocessing & 4 RR features (Equations for pre-RR, post-RR, local context, ratio).
│   - Ensemble architecture & class-weighted objective.
│   - Post-hoc Temperature Scaling calibration.
│   - 3D Uncertainty Framework & Cluster-Based Entropy formulation.
├── IV. Experimental Setup:
│   - MIT-BIH database & de Chazal DS1/DS2 inter-patient protocol.
│   - Table I (Dataset partition statistics).
├── V. Results & Discussion:
│   - A. Per-Class Diagnostic Performance (Table II & Figure 2 Confusion Matrix).
│   - B. Ablation on RR Features and Ensembling (Table III).
│   - C. The Data Leakage Gap: Beat-wise vs. Inter-patient (Table IV).
│   - D. Reliability & Probability Calibration (Table V).
│   - E. Selective Classification & Cardiologist Referral Sweeps (Table VI, VII & Figure 1 ARC Curve).
├── VI. Clinical Decision Support System:
│   - Brief discussion of real-time telemetry HUD and automated cardiologist referral protocol.
└── VII. Conclusion:
    - Summary of safety-aware ECG classification and future multi-lead validation.
```

---

## 🎯 5. Recommended Submission Venues

1. **IEEE EMBC** (Engineering in Medicine and Biology Conference) — Premier international conference for biomedical engineering.
2. **IEEE BHI** (Biomedical and Health Informatics) — Specializes in clinical AI, telemetry, and healthcare systems.
3. **IEEE CBMS** (Computer-Based Medical Systems) — Focuses on medical software, safety protocols, and decision support.
4. **IEEE MeMeA** (Medical Measurements and Applications) — Emphasizes measurement reliability, calibration, and sensor telemetry.
5. **IEEE Access** (Open-Access Journal) — Good target for a full-length 8–10 page journal paper after conference review.
