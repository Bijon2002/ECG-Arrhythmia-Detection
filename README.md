# ECG Arrhythmia Detection

A complete deep learning pipeline for detecting and classifying ECG arrhythmias, using PyTorch.

## Project Roadmap & Status

### PHASE 1: PROJECT SETUP
- [x] **Step 1:** Created complete folder structure (`config/`, `data/`, `models/`, `results/`, `src/`, `notebooks/`, `tests/`)
- [x] **Step 2:** Created configuration system (`config/config.py`)
- [x] **Step 3:** Created sample ECG file (`data/samples/normal_sample.csv`)
- [x] **Step 4:** Created utilities module (`src/utils.py`)
- [x] **Step 5:** Tested complete setup

### PHASE 2: DATA ACQUISITION & PROCESSING
- [x] **Step 6:** Download MIT-BIH ECG data
- [ ] **Step 7:** Process ECG signals

### PHASE 3: MODELING
- [ ] **Step 8:** Build CNN model
- [ ] **Step 9:** Train ensemble
- [ ] **Step 10-12:** Uncertainty methods

### PHASE 4: DEPLOYMENT
- [ ] **Step 13-15:** Evaluation & web app

## Environment Setup

The project uses a Conda virtual environment with the following key packages:
- Python 3.10
- PyTorch >= 2.0.0
- wfdb >= 4.1.0
- Streamlit >= 1.28.0

To recreate the environment:
```bash
conda create -n ecg_env python=3.10
conda activate ecg_env
pip install -r requirements.txt
```

## Usage

### 1. Data Acquisition
To download the MIT-BIH Arrhythmia Database (~72 MB) into the `data/raw/` directory:
```bash
python src/download_data.py
```

