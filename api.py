import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

print("Loading 5 independently trained AI brains + Calibration Temperatures...")
for i in range(1, 6):
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

@app.route('/hints')
def serve_hints():
    """Serves the hidden hints.html for defense reference."""
    return send_from_directory(app.static_folder, 'hints.html')

@app.route('/save_viva_question', methods=['POST'])
def save_viva_question():
    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400
        
    viva_file = os.path.join(BASE_DIR, 'viva_prep.md')
    
    # Create file with a header if it doesn't exist
    if not os.path.exists(viva_file):
        with open(viva_file, 'w') as f:
            f.write("# Viva Preparation Questions\n\n")
            
    with open(viva_file, 'a') as f:
        f.write(f"- **Q:** {question}\n")
        
    return jsonify({"success": "Question saved to viva_prep.md!"})

@app.route('/random_beat/<beat_type>', methods=['GET'])
def get_random_beat(beat_type):
    """Pulls a random heartbeat from our test set! Supports normal, abnormal, N, S, V, F, Q, and noisy."""
    test_data = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'test_data.npz'))
    X_test = test_data['X']
    y_test = test_data['y']
    
    beat_type_lower = beat_type.lower()
    if beat_type_lower in ['normal', 'n', '0']:
        indices = np.where(y_test == 0)[0]
    elif beat_type_lower in ['abnormal']:
        indices = np.where(y_test > 0)[0]
    elif beat_type_lower in ['s', 'sveb', '1']:
        indices = np.where(y_test == 1)[0]
    elif beat_type_lower in ['v', 'veb', 'pvc', '2']:
        indices = np.where(y_test == 2)[0]
    elif beat_type_lower in ['f', 'fusion', '3']:
        indices = np.where(y_test == 3)[0]
    elif beat_type_lower in ['q', 'paced', 'unknown', '4']:
        indices = np.where(y_test == 4)[0]
    elif beat_type_lower in ['noisy', 'artifact', 'ood']:
        indices = np.where(y_test == 0)[0]
        random_idx = np.random.choice(indices)
        clean_sig = X_test[random_idx].copy()
        t = np.linspace(0, 1, len(clean_sig))
        noise = 0.45 * np.random.randn(len(clean_sig)) + 0.55 * np.sin(2 * np.pi * 3.5 * t)
        noisy_sig = (clean_sig + noise).tolist()
        return jsonify({
            "signal": noisy_sig,
            "true_diagnosis": "Electrode Artifact / Noisy OOD Signal",
            "class_code": "Artifact"
        })
    else:
        indices = np.arange(len(y_test))
        
    random_idx = np.random.choice(indices)
    signal = X_test[random_idx].tolist()
    true_label = int(y_test[random_idx])
    
    true_diagnosis, _ = CLASS_MAPPING.get(true_label, ("Unknown", ""))
    
    return jsonify({
        "signal": signal,
        "true_diagnosis": true_diagnosis,
        "class_code": ["N", "S", "V", "F", "Q"][true_label] if true_label < 5 else "Q"
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
    individual_models = []
    mc_dropout_predictions = []
    
    with torch.no_grad():
        for idx, (model, temp) in enumerate(zip(ensemble_models, ensemble_temperatures)):
            # Standard Evaluation (Dropout OFF)
            model.eval()
            logits = model(tensor_signal)
            calibrated_logits = logits / temp
            probs = torch.softmax(calibrated_logits, dim=1)
            probs_np = probs.numpy()[0]
            ensemble_predictions.append(probs_np)
            
            m_pred_class = int(np.argmax(probs_np))
            m_diag, m_sev = CLASS_MAPPING.get(m_pred_class, ("Unknown", "unknown"))
            individual_models.append({
                "model_id": idx + 1,
                "predicted_class": m_pred_class,
                "diagnosis": m_diag,
                "severity": m_sev,
                "confidence": float(probs_np[m_pred_class] * 100),
                "temperature": float(temp),
                "probs": [float(p) for p in probs_np]
            })
            
            # MC Dropout Evaluation (Dropout ON)
            enable_dropout(model)
            for _ in range(3): # 3 passes per model
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
        "is_uncertain": is_uncertain,
        "individual_models": individual_models
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

@app.route('/batch_predict/auto', methods=['POST'])
def batch_predict_auto():
    try:
        test_data = np.load(os.path.join(BASE_DIR, 'data', 'processed', 'test_data.npz'))
        X_test = test_data['X']
        y_test = test_data['y']
        
        normal_indices = np.where(y_test == 0)[0]
        abnormal_indices = np.where(y_test > 0)[0]
        
        # Grab 25 of each
        selected_normals = np.random.choice(normal_indices, 25, replace=False)
        selected_abnormals = np.random.choice(abnormal_indices, 25, replace=False)
        all_indices = np.concatenate([selected_normals, selected_abnormals])
        np.random.shuffle(all_indices)
        
        results = []
        for idx, i in enumerate(all_indices):
            signal = X_test[i][:360]
            tensor_signal = torch.tensor(signal, dtype=torch.float32).view(1, 1, 360)
            
            ensemble_predictions = []
            mc_dropout_predictions = []
            
            with torch.no_grad():
                for model, temp in zip(ensemble_models, ensemble_temperatures):
                    model.eval()
                    logits = model(tensor_signal)
                    probs = torch.softmax(logits / temp, dim=1)
                    ensemble_predictions.append(probs.numpy())
                    
                    enable_dropout(model)
                    for _ in range(3):
                        mc_logits = model(tensor_signal)
                        mc_probs = torch.softmax(mc_logits / temp, dim=1)
                        mc_dropout_predictions.append(mc_probs.numpy())
            
            mean_probs = np.mean(np.vstack(ensemble_predictions), axis=0)
            mc_dropout_uncertainty = float(np.mean(np.var(np.vstack(mc_dropout_predictions), axis=0)))
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
