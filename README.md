<div align="center">

# 🫀 High-Confidence ECG Arrhythmia Classification & Uncertainty Quantification System

### *A Clinically-Calibrated True Deep Ensemble with Latent Topology & 3D Uncertainty Gauges*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PhysioNet](https://img.shields.io/badge/Dataset-MIT--BIH%20PhysioNet-0284c7?style=for-the-badge)](https://physionet.org/content/mitdb/1.0.0/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.4%25-10b981?style=for-the-badge)](#-empirical-benchmarks--results)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-96.0%25-6366f1?style=for-the-badge)](#-empirical-benchmarks--results)

<br/>

**Author:** [Bijosilin Marisilin](https://github.com/Bijon2002) (Student ID: `2541518`)  
**Supervisor:** Ms. Niroji Thayalan  
**Course:** BSc (Hons) Computer Science & Software Engineering  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT) Northern Uni / University of Bedfordshire (UK)

---

</div>

## 📌 Executive Summary & Clinical Motivation

Standard Deep Learning classifiers for automated Electrocardiogram (ECG) interpretation achieve high benchmark accuracy but suffer from a fatal clinical defect: **deterministic overconfidence**. When presented with ambiguous heartbeats, rare morphologies (e.g., Fusion beats), or Out-of-Distribution (OOD) sensor noise, standard neural networks output dangerously high confidence probabilities on erroneous predictions.

This project delivers an end-to-end, safety-critical medical telemetry system that solves this black-box limitation. By pairing a **5-Model True Deep Ensemble** with **Post-hoc Temperature Scaling** and a novel **3D Uncertainty Framework** (Epistemic MC Dropout, Aleatoric Shannon Entropy, and Latent **Cluster-Based Entropy**), the platform accurately classifies cardiac rhythms while autonomously refusing low-confidence predictions to protect patient safety.

---

## 🌟 Key Features & Innovations

* 🧠 **5-Member True Deep Ensemble:** Five independently initialized 1D-CNN architectures trained with diverse stochastic seeds (Seeds `42, 101, 202, 303, 404`) to capture epistemic model variance.
* 🎯 **Post-hoc Temperature Scaling:** Minimizes Expected Calibration Error (ECE from **0.041 $\to$ 0.012**, a **70.7% reduction**) ensuring output probabilities represent true clinical likelihoods.
* 🔬 **Novel Cluster-Based Entropy (CBE):** Formulates clinical risk clusters ($\{N\}, \{S\}, \{V, F\}, \{Q\}$) to suppress benign intra-cluster confusion while selectively flagging high-risk patient ambiguities.
* 🛡️ **OOD & Noise Safety Alarm:** Dynamic uncertainty thresholds trigger automated cardiologist referral protocols whenever signal corruption or out-of-distribution patterns are detected.
* ⚡ **Real-Time Clinical Telemetry HUD:** High-framerate single-lead ECG oscilloscope (360 Hz), live audio tone generation, dynamic BPM tracker, and 50-beat concurrent Holter batch processor.

---

## 🏛️ System Architecture

```
                                  MIT-BIH Arrhythmia Dataset (109,441 Beats)
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          ▼                                                     ▼
              [Butterworth 0.5-45 Hz]                              [Pan-Tompkins QRS Detection]
              (Removes Drift & Noise)                              (360-Sample Beat Windowing)
                          └──────────────────────────┬──────────────────────────┘
                                                     ▼
                                        [Z-Score Per-Beat Normalization]
                                                     │
                ┌────────────────────────────────────┼────────────────────────────────────┐
                ▼                                    ▼                                    ▼
       ┌───────────────────┐                ┌───────────────────┐                ┌───────────────────┐
       │ 1D-CNN (Seed 42)  │                │ 1D-CNN (Seed 101) │                │ 1D-CNN (Seed 404) │
       │ [3x Conv + Linear]│                │ [3x Conv + Linear]│                │ [3x Conv + Linear]│
       └─────────┬─────────┘                └─────────┬─────────┘                └─────────┬─────────┘
                 │                                    │                                    │
                 └────────────────────────────────────┼────────────────────────────────────┘
                                                      ▼
                                       [Temperature Scaling Calibration]
                                                      │
                                                      ▼
                                    [Ensemble Aggregation & 3D UQ Engine]
                                      ├── Epistemic Variance (MC Dropout)
                                      ├── Aleatoric Predictive Entropy
                                      └── Latent Cluster-Based Entropy (CBE)
                                                      │
                                                      ▼
                                ┌───────────────────────────────────────────┐
                                │      Clinical Telemetry Web Dashboard     │
                                │   (Safe 🟢  |  Moderate 🟠  |  High-Risk 🔴)   │
                                └───────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks & Results

Evaluated across the official test split of the **MIT-BIH Arrhythmia Database** (AAMI EC57 standard):

| Metric | Single Baseline 1D-CNN | 5-Member Deep Ensemble (Ours) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 94.20% | **98.40%** | **+4.20%** |
| **Macro F1-Score** | 87.60% | **96.00%** | **+8.40%** |
| **Normal (N) F1** | 97.50% | **99.20%** | +1.70% |
| **Supraventricular (S) F1** | 82.10% | **91.50%** | **+9.40%** |
| **Ventricular Ectopic (V) F1**| 90.10% | **97.40%** | **+7.30%** |
| **Fusion (F) F1** | 68.40% | **88.60%** | **+20.20%** |
| **Unknown / Paced (Q) F1** | 91.00% | **98.10%** | +7.10% |
| **Expected Calibration Error (ECE)** | 0.0410 | **0.0120** | **-70.7% (High Calibration)** |
| **Inference Latency** | 6 ms | **<25 ms** | Real-Time Telemetry Capable |

---

## 🗂️ Project Directory Structure

```text
ECG-Arrhythmia-Detection/
├── api.py                          # Production Flask REST API backend
├── app.py                          # Interactive terminal inference & testing engine
├── config/
│   └── config.py                   # Central hyperparameters, signal constants & paths
├── data/
│   ├── raw/                        # Raw PhysioNet MIT-BIH records (.dat, .hea, .atr)
│   └── processed/                  # Segmented 360-sample heartbeat tensors & labels
├── frontend/
│   ├── index.html                  # Clinical Cardiologist HUD & Telemetry Dashboard
│   ├── app.js                      # WebSocket / REST polling & Chart.js engine
│   ├── style.css                   # Responsive dark/light medical design system
│   ├── poster.html                 # 4K Ultra-HD Academic Research Poster
│   └── hints.html                  # Clinical triage guide & uncertainty manual
├── models/
│   ├── ensemble_model_1.pth        # Ensemble Member 1 (Seed 42)
│   ├── ensemble_model_2.pth        # Ensemble Member 2 (Seed 101)
│   ├── ensemble_model_3.pth        # Ensemble Member 3 (Seed 202)
│   ├── ensemble_model_4.pth        # Ensemble Member 4 (Seed 303)
│   ├── ensemble_model_5.pth        # Ensemble Member 5 (Seed 404)
│   └── ecg_model_v1.pth            # Single baseline checkpoint
├── results/
│   ├── logs/                       # Training histories, calibration & MC Dropout logs
│   ├── metrics/                    # Confusion matrices, classification reports, ECE scores
│   └── plots/                      # High-resolution thesis figures & reliability plots
├── src/
│   ├── download_data.py            # Automated MIT-BIH dataset ingestion
│   ├── preprocess.py               # Butterworth filtering, Pan-Tompkins, Z-score scaling
│   ├── model.py                    # PyTorch 1D-CNN neural network architecture
│   ├── train_ensemble.py           # Multi-seed Deep Ensemble training pipeline
│   ├── calibrate.py                # Post-hoc Temperature Scaling optimizer
│   ├── cbe_engine.py               # Latent space Cluster-Based Entropy algorithm
│   └── evaluate.py                 # Multi-beat uncertainty testing suite
├── UG_Project_Setup_Guidance.md    # Formal undergraduate project setup manual
├── requirements.txt                # Python package dependency manifest
└── README.md                       # Main project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository & Setup Environment
```bash
# Clone repository
git clone https://github.com/Bijon2002/ECG-Arrhythmia-Detection.git
cd ECG-Arrhythmia-Detection

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# (Linux / macOS)
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Start the Backend REST API Server
```bash
python api.py
```
*The server boots the 5 ensemble models, initializes calibrated temperature parameters, and listens on `http://127.0.0.1:5000`.*

---

### 3. Launch the Clinical Dashboard (HUD)
* **Direct Launch:** Double-click `frontend/index.html` or open it with Google Chrome.
* **Or via Local Server:**
```bash
cd frontend
python -m http.server 5500
```
Navigate to: `http://127.0.0.1:5500`

---

## 🧪 Reproducing Experiments & Pipelines

```bash
# 1. Download full MIT-BIH dataset (48 records, ~72 MB)
python src/download_data.py

# 2. Run signal filtering, QRS beat extraction & train/test partitioning
python src/preprocess.py

# 3. Train all 5 True Deep Ensemble members
python src/train_ensemble.py

# 4. Compute optimal Temperature Scaling parameter (T) for calibration
python src/calibrate.py

# 5. Run Monte Carlo Dropout & generate uncertainty validation logs
python src/evaluate.py
```

---

## 📡 REST API Documentation

### `POST /predict`
Submits a 360-sample normalized single-beat array for multi-model ensemble classification and 3D uncertainty estimation.

* **Request Body:**
  ```json
  {
    "signal": [0.012, 0.045, -0.018, "... (360 float values) ..."]
  }
  ```
* **Response (HTTP 200):**
  ```json
  {
    "predicted_class": "V",
    "class_name": "Ventricular Ectopic Beat (VEB)",
    "confidence": 0.9842,
    "calibrated_probabilities": {
      "N": 0.0051,
      "S": 0.0042,
      "V": 0.9842,
      "F": 0.0045,
      "Q": 0.0020
    },
    "uncertainty_metrics": {
      "epistemic_variance": 0.0004,
      "predictive_entropy": 0.0821,
      "cluster_based_entropy": 0.0112
    },
    "clinical_safety": {
      "flag": "SAFE",
      "action": "AUTOMATED_TRIAGE_APPROVED"
    }
  }
  ```

### `GET /random_beat`
Returns an annotated test heartbeat from the MIT-BIH dataset with its ground truth label for interactive evaluation.

---

## 🎓 Academic Citations & Project Credits

This project was developed as a final year undergraduate research project at the **University of Bedfordshire (UK)** in partnership with **SLIIT Northern Uni**.

```bibtex
@bachelorsthesis{marisilin2026ecg,
  author       = {Bijosilin Marisilin},
  title        = {Uncertainty-Aware True Deep Ensemble for ECG Arrhythmia Classification and Clinical Decision Support},
  school       = {University of Bedfordshire / SLIIT Northern Uni},
  year         = {2026},
  month        = {August},
  note         = {Unit: CIS017-3 Undergraduate Project}
}
```

---

## 📄 License
This project is released under the [MIT License](LICENSE). Built for academic, research, and non-commercial clinical validation purposes.
