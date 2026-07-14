import os
import sys
import torch
import numpy as np
import wfdb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Setup path to import src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModel
from uncertainty import calculate_predictive_entropy, calculate_cluster_based_entropy

# --- FLASK CONFIGURATION ---
# We configure Flask to serve our static HTML frontend automatically!
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

print("Starting True Ensemble Flask API...")

# --- LOAD DEEP ENSEMBLE ---
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ensemble_models = []
ensemble_temperatures = []

print("Loading 3 independently trained AI brains + Calibration Temperatures...")
for i in range(1, 4):
    # Load Model
    model = ECGModel(num_classes=5)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'ecg_model_v{i}.pth'), map_location="cpu", weights_only=True))
    model.eval()
    ensemble_models.append(model)
    
    # Load Temperature
    temp_path = os.path.join(MODELS_DIR, f'temperature_v{i}.txt')
    if os.path.exists(temp_path):
        with open(temp_path, 'r') as f:
            t = float(f.read().strip())
    else:
        t = 1.0 # Default if not calibrated yet
    ensemble_temperatures.append(t)

CLASS_MAPPING = {
    0: ("Normal Beat", "normal"),
    1: ("Supraventricular Ectopic", "warning"),
    2: ("Ventricular Ectopic", "danger"),
    3: ("Fusion Beat", "warning"),
    4: ("Unknown Beat", "unknown")
}

# --- ROUTES ---

@app.route('/')
def serve_frontend():
    """Serves the index.html from the frontend folder."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/random_beat', methods=['GET'])
def get_random_beat():
    """Pulls a random heartbeat from our massive 21,888 test set!"""
    test_data = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'test_data.npz'))
    X_test = test_data['X']
    y_test = test_data['y']
    
    random_idx = np.random.randint(0, len(X_test))
    signal = X_test[random_idx].tolist()
    true_label = int(y_test[random_idx])
    
    true_diagnosis, _ = CLASS_MAPPING.get(true_label, ("Unknown", ""))
    
    return jsonify({
        "signal": signal,
        "true_diagnosis": true_diagnosis
    })

@app.route('/patient_stream/<int:start_idx>', methods=['GET'])
def get_patient_stream(start_idx):
    """Streams actual raw ECG data sequentially from Patient 200."""
    try:
        # Read a 5-second chunk (1800 samples) to ensure we always have buffer
        chunk_size = 1800
        record_path = os.path.join(BASE_DIR, 'data', 'raw', '200')
        
        record = wfdb.rdrecord(record_path, sampfrom=start_idx, sampto=start_idx + chunk_size)
        signal = record.p_signal[:, 0]
        
        # Simple normalization to [-1, 1] range for visualization
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        return jsonify({
            "signal": signal.tolist(),
            "true_diagnosis": "Raw Patient 200 Stream"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'signal' not in data:
        return jsonify({"error": "No signal provided."}), 400
        
    signal = data['signal']
    if len(signal) != 360:
        return jsonify({"error": f"Signal must contain exactly 360 samples, got {len(signal)}."}), 400
        
    tensor_signal = torch.tensor(signal, dtype=torch.float32).view(1, 1, 360)
    
    # Run predictions across the True Deep Ensemble
    ensemble_predictions = []
    
    # 1. MC Dropout for Epistemic Uncertainty (As requested in report)
    # We enable dropout and do multiple forward passes
    mc_dropout_predictions = []
    
    with torch.no_grad():
        for model, temp in zip(ensemble_models, ensemble_temperatures):
            # Standard Evaluation (Dropout OFF)
            model.eval()
            logits = model(tensor_signal)
            calibrated_logits = logits / temp
            probs = torch.softmax(calibrated_logits, dim=1)
            ensemble_predictions.append(probs.numpy())
            
            # MC Dropout Evaluation (Dropout ON)
            enable_dropout(model)
            for _ in range(3): # 3 passes per model = 9 total MC passes
                mc_logits = model(tensor_signal)
                mc_probs = torch.softmax(mc_logits / temp, dim=1)
                mc_dropout_predictions.append(mc_probs.numpy())
            
    # Combine predictions
    ensemble_predictions = np.vstack(ensemble_predictions)
    mean_probs = np.mean(ensemble_predictions, axis=0)
    
    # Calculate MC Dropout Epistemic Uncertainty (Variance across passes)
    mc_dropout_predictions = np.vstack(mc_dropout_predictions)
    mc_dropout_uncertainty = float(np.mean(np.var(mc_dropout_predictions, axis=0)))
    
    # Calculate Advanced Uncertainties (as requested in the report)
    pred_entropy = calculate_predictive_entropy(mean_probs)
    cluster_entropy = calculate_cluster_based_entropy(mean_probs)
    
    predicted_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[predicted_class] * 100)
    
    diagnosis_text, severity = CLASS_MAPPING.get(predicted_class, ("Unknown", "unknown"))
    
    is_uncertain = cluster_entropy > 0.5 
    
    return jsonify({
        "diagnosis": diagnosis_text,
        "severity": severity,
        "confidence": confidence,
        "predictive_entropy": pred_entropy,
        "cluster_entropy": cluster_entropy,
        "mc_dropout_uncertainty": mc_dropout_uncertainty,
        "is_uncertain": is_uncertain
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        results = []
        
        for idx, line in enumerate(lines):
            # Skip empty lines or header if present (assuming numeric data only for simplicity)
            if not line.strip(): continue
            try:
                values = [float(x) for x in line.split(',')]
            except ValueError:
                continue # Skip non-numeric header lines
                
            if len(values) < 360:
                continue # Skip invalid lines
                
            signal = values[:360]
            tensor_signal = torch.tensor(signal, dtype=torch.float32).view(1, 1, 360)
            
            ensemble_predictions = []
            mc_dropout_predictions = []
            
            with torch.no_grad():
                for model, temp in zip(ensemble_models, ensemble_temperatures):
                    model.eval()
                    logits = model(tensor_signal)
                    calibrated_logits = logits / temp
                    probs = torch.softmax(calibrated_logits, dim=1)
                    ensemble_predictions.append(probs.numpy())
                    
                    enable_dropout(model)
                    for _ in range(3):
                        mc_logits = model(tensor_signal)
                        mc_probs = torch.softmax(mc_logits / temp, dim=1)
                        mc_dropout_predictions.append(mc_probs.numpy())
            
            ensemble_predictions = np.vstack(ensemble_predictions)
            mean_probs = np.mean(ensemble_predictions, axis=0)
            
            mc_dropout_predictions = np.vstack(mc_dropout_predictions)
            mc_dropout_uncertainty = float(np.mean(np.var(mc_dropout_predictions, axis=0)))
            
            pred_entropy = calculate_predictive_entropy(mean_probs)
            cluster_entropy = calculate_cluster_based_entropy(mean_probs)
            
            predicted_class = int(np.argmax(mean_probs))
            confidence = float(mean_probs[predicted_class] * 100)
            diagnosis_text, severity = CLASS_MAPPING.get(predicted_class, ("Unknown", "unknown"))
            is_uncertain = cluster_entropy > 0.5 
            
            results.append({
                "index": idx + 1,
                "diagnosis": diagnosis_text,
                "severity": severity,
                "confidence": confidence,
                "predictive_entropy": pred_entropy,
                "cluster_entropy": cluster_entropy,
                "mc_dropout_uncertainty": mc_dropout_uncertainty,
                "is_uncertain": is_uncertain
            })
            
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
