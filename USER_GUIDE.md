# ECG Arrhythmia Detection System - User Guide

## 1. Prerequisites
Ensure you have the following installed on your system:
*   Anaconda or Miniconda
*   Python 3.10
*   A modern web browser (Chrome/Edge/Firefox)

## 2. Environment Setup
To reproduce the exact environment used for this project, open your Anaconda Prompt (or terminal) and execute the following commands in the project root directory (`f:\ECG-Arrhythmia-Detection`):

```bash
# Create the conda environment
conda create -n ecg_env python=3.10 -y

# Activate the environment
conda activate ecg_env

# Install the required dependencies
pip install -r requirements.txt
```

## 3. Running the Backend Server
The Flask REST API must be running to serve predictions to the dashboard. 

```bash
# Ensure you are in the project root directory
cd /path/to/ECG-Arrhythmia-Detection

# Start the Flask API
python app.py
```
You should see output indicating that the server is running on `http://127.0.0.1:8000`. Leave this terminal window open.

## 4. Accessing the Clinical Dashboard
The dashboard is a static HTML/JS frontend that communicates with the Flask backend.
1. Open your File Explorer.
2. Navigate to the `frontend` folder inside the project directory.
3. Double-click `index.html` to open it in your web browser.

## 5. Using the Dashboard
*   **Random Beat Testing:** Click the "Test Random Normal Beat" or "Test Random Abnormal Beat" buttons. This will fetch a random heartbeat from the test set, display its waveform, and show the model's prediction alongside the uncertainty metrics.
*   **Uncertainty Flags:** Pay attention to the flag on the right. If the AI is uncertain (Cluster-Based Entropy > 0.5), it will flag RED for clinical review.
*   **Batch Upload:** You can use the "Upload CSV" feature to process multiple heartbeats at once.

## 6. Project Structure Overview
*   `/src`: Contains the source code for preprocessing, model definition, training, and uncertainty quantification.
*   `/models`: Contains the saved PyTorch `.pth` model weights and temperature scaling `.txt` files for the 5 ensemble members.
*   `/results`: Contains the evaluation logs, metrics, and generated figures.
*   `/frontend`: Contains the web dashboard files (`index.html`, `style.css`, `main.js`).
*   `app.py`: The main Flask server application.
*   `api.py`: The API routing and inference logic.
