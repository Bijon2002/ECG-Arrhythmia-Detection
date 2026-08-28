# High-Confidence ECG Arrhythmia Classification & Uncertainty Quantification System Using True Deep Ensembles and Latent Space Topology

**Student Name:** Bijosilin Marisilin  
**Student ID:** 2541518  
**Unit:** CIS017-3 Undergraduate Project  
**Course:** BSc (Hons) Computer Science & Software Engineering  
**Institution:** University of Bedfordshire / SLIIT Northern Uni  

---

## UG Project Setup Guidance

### 1. Overview
This project comprises a production-ready, clinical-grade AI diagnostic telemetry platform with three core components:
1. **Deep Learning Core & Backend API:** PyTorch-powered True Deep Ensemble (5 independent 1D-CNNs), Temperature Scaling calibrator, Cluster-Based Entropy (CBE) uncertainty engine, and Flask REST API backend.
2. **Clinical Telemetry HUD (Frontend):** Interactive web dashboard providing real-time single-lead ECG wave visualization, 5-class AAMI arrhythmia classification, 3D uncertainty breakdown, and batch Holter report generation.
3. **Data Preprocessing & Evaluation Pipeline:** Automated MIT-BIH dataset ingestion, Butterworth bandpass filtering (0.5–45 Hz), Pan-Tompkins QRS beat segmentation, and Out-of-Distribution (OOD) stress testing.

---

### 2. Project Directory Structure
```text
ECG-Arrhythmia-Detection (2541518)/
├── api.py                          # Main Flask REST API server (port 5000)
├── app.py                          # Interactive CLI / Local inference engine
├── config/
│   └── config.py                   # Global hyperparameters, paths & AAMI class mappings
├── data/
│   ├── raw/                        # PhysioNet MIT-BIH raw recordings (.dat, .hea, .atr)
│   └── processed/                  # Segmented & normalized 360-sample heartbeat tensors
├── frontend/
│   ├── index.html                  # Clinical Cardiologist HUD & Telemetry Dashboard
│   ├── app.js                      # Real-time WebSocket/REST polling & Chart.js engine
│   ├── style.css                   # Medical HUD dark/light responsive interface styles
│   ├── poster.html                 # 4K Ultra-HD Academic Research Poster
│   └── hints.html                  # Clinical triage guide & uncertainty interpretability manual
├── models/
│   ├── ensemble_model_1.pth        # Trained PyTorch Ensemble Member 1
│   ├── ensemble_model_2.pth        # Trained PyTorch Ensemble Member 2
│   ├── ensemble_model_3.pth        # Trained PyTorch Ensemble Member 3
│   ├── ensemble_model_4.pth        # Trained PyTorch Ensemble Member 4
│   ├── ensemble_model_5.pth        # Trained PyTorch Ensemble Member 5
│   └── ecg_model_v1.pth            # Baseline 1D-CNN model checkpoint
├── results/
│   ├── logs/                       # Training, calibration, and MC Dropout evaluation logs
│   ├── metrics/                    # Confusion matrices, Macro F1, and ECE scores
│   └── plots/                      # Publication-grade figures & reliability diagrams
├── src/
│   ├── download_data.py            # Automated PhysioNet MIT-BIH dataset downloader
│   ├── preprocess.py               # Butterworth filtering, Pan-Tompkins, Z-score scaling
│   ├── model.py                    # PyTorch 1D-CNN neural network architecture
│   ├── train_ensemble.py           # 5-seed True Deep Ensemble training routine
│   ├── calibrate.py                # Post-hoc Temperature Scaling optimizer (ECE reduction)
│   ├── cbe_engine.py               # Latent space Cluster-Based Entropy uncertainty extractor
│   └── evaluate.py                 # Multi-beat uncertainty testing & Monte Carlo Dropout
├── requirements.txt                # Python package dependency manifest
└── README.md                       # High-level architecture documentation
```

---

### 3. Backend & AI Core Setup Instructions

#### Requirements
* Python 3.9, 3.10, or 3.11 (64-bit recommended)
* PyCharm, VS Code, or Cursor IDE
* Git for version control

#### Steps
1. **Open Project Folder:**
   Open the root `ECG-Arrhythmia-Detection` directory in your IDE.

2. **Create & Activate a Python Virtual Environment:**
   Open your terminal (PowerShell / Command Prompt) in the project root:
   ```powershell
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment (Windows PowerShell)
   .\venv\Scripts\Activate.ps1

   # (If using CMD)
   .\venv\Scripts\activate.bat
   ```

3. **Install Dependencies:**
   Install all required scientific and deep learning packages:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Core libraries installed: `torch`, `torchvision`, `torchaudio`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `flask`, `flask-cors`, `wfdb`, `pandas`)*

4. **Verify Data & Pretrained Weights:**
   Ensure the 5 ensemble model weights exist in the `models/` directory:
   * `models/ensemble_model_1.pth` through `models/ensemble_model_5.pth`

5. **Start the Flask Backend REST API:**
   Launch the REST API server:
   ```powershell
   python api.py
   ```
   *The backend will initialize the 5 neural network models, calibrate temperature parameters, and start listening on `http://127.0.0.1:5000`.*

---

### 4. Frontend & Telemetry HUD Setup Instructions

#### Requirements
* Any modern web browser (Google Chrome, Microsoft Edge, Brave, or Mozilla Firefox)
* Optional: VS Code *Live Server* extension or Python HTTP server

#### Steps
1. **Option A (Direct Browser Launch - Recommended):**
   * Double-click or open `frontend/index.html` directly in your web browser.
   * Or right-click `frontend/index.html` $\to$ **Open with Google Chrome**.

2. **Option B (Local Web Server):**
   * If running from VS Code, right-click `frontend/index.html` $\to$ **Open with Live Server** (Port `5500`).
   * Or start a quick Python web server from the `frontend/` folder:
     ```powershell
     cd frontend
     python -m http.server 5500
     ```
     Navigate to: `http://127.0.0.1:5500`

3. **Verify API Connection Status:**
   * Look at the top-right status indicator in the HUD:
     * 🟢 **"REST API: Online (Port 5000)"** confirms full end-to-end communication.

---

### 5. Running Experiments & Re-evaluation (Optional Workflows)

#### A. Download Raw MIT-BIH Data & Preprocess:
```powershell
# Ingest 48 PhysioNet patient records
python src/download_data.py

# Execute 0.5-45 Hz filtering, Pan-Tompkins QRS detection & AAMI 5-class windowing
python src/preprocess.py
```

#### B. Train the 5-Member True Deep Ensemble:
```powershell
# Trains 5 independent CNNs with distinct random seeds (Seeds 42, 101, 202, 303, 404)
python src/train_ensemble.py
```

#### C. Perform Post-hoc Temperature Scaling Calibration:
```powershell
# Optimizes temperature T on validation set to minimize Expected Calibration Error (ECE)
python src/calibrate.py
```

#### D. Run MC Dropout & Uncertainty Evaluation:
```powershell
# Generates predictive variance logs and exports results to results/logs/evaluation_log.txt
python src/evaluate.py
```

---

### 6. End-to-End System Integration & Demonstration Workflow

To execute the complete live system demonstration for assessors:

1. **Step 1:** Start the Python backend API server:
   ```powershell
   python api.py
   ```
2. **Step 2:** Open `frontend/index.html` in Google Chrome.
3. **Step 3 (Single-Beat Telemetry):**
   * Click **"Random Beat"** to fetch a patient heartbeat from the MIT-BIH test partition.
   * Click **"Diagnose ECG"** $\to$ observe the 5-member ensemble vote, confidence breakdown, predictive entropy, and the safety triage status (Green / Amber / Red).
4. **Step 4 (Corrupted ECG / OOD Stress Test):**
   * Toggle **"Inject High Noise (VIVA Out-of-Distribution)"**.
   * Click **"Diagnose ECG"** $\to$ observe how the system prevents black-box overconfidence by firing the **⚠️ HIGH UNCERTAINTY / CARDIOLOGIST REVIEW REQUIRED** clinical safety alarm.
5. **Step 5 (Batch Holter Report):**
   * Switch to the **"Batch Holter Engine"** tab $\to$ click **"Process 100 Consecutive Beats"** $\to$ review the automated patient triage summary chart and printable PDF report.

---

### 7. Troubleshooting & Important Notes
* **Port Conflict on Port 5000:** If port 5000 is occupied by another application, edit `api.py` line 260 to change `port=5000` to `port=5001`, and update `API_BASE_URL` in `frontend/app.js` line 12.
* **CORS Errors:** Flask-CORS is pre-configured in `api.py` (`CORS(app)`) to permit seamless local file requests (`file:///`) and localhost origins.
* **Model Checkpoints:** Do not rename or delete `.pth` files in `models/` as the ensemble loader validates all 5 seeds on boot.
* **CPU vs GPU Execution:** The system automatically detects CUDA-compatible GPUs; if no GPU is available, it gracefully defaults to high-speed multi-threaded CPU inference (<25ms per beat).
