import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import ECGModel

# Bulletproof Absolute Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def load_test_data():
    print("Loading unseen testing heartbeats...")
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    
    # Format shape for PyTorch
    X_test = torch.tensor(test_data['X'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data['y'], dtype=torch.long)
    
    test_dataset = TensorDataset(X_test, y_test)
    # Batch size 1 so we can easily evaluate uncertainty for one heartbeat at a time
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

def enable_dropout(model):
    """
    THE PYTORCH TRICK:
    This function forces the Dropout layer to stay ON during testing 
    so we can create our "Virtual Ensemble" using Monte Carlo Dropout.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def evaluate_uncertainty(num_samples_to_test=50, mc_iterations=10):
    test_loader = load_test_data()
    
    # 1. Load the trained AI Brain
    model = ECGModel(num_classes=5)
    model_path = os.path.join(MODELS_DIR, 'ecg_model_v1.pth')
    model.load_state_dict(torch.load(model_path))
    
    # 2. Put model in eval mode, but forcefully turn Dropout BACK ON
    model.eval()
    enable_dropout(model)
    
    log_file_path = os.path.join(RESULTS_DIR, 'evaluation_log.txt')
    uncertain_count = 0
    
    # Open our log file for permanent recording
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write("ECG Arrhythmia Uncertainty Evaluation Log (MC Dropout)\n")
        log_file.write("====================================================\n\n")
        
        print(f"\nTesting {num_samples_to_test} heartbeats using {mc_iterations} MC Dropout iterations...")
        
        # Look at one heartbeat at a time
        for idx, (data, true_label) in enumerate(test_loader):
            if idx >= num_samples_to_test:
                break
                
            # 3. Ask the brain to diagnose the SAME heartbeat 10 times in a row
            mc_predictions = []
            with torch.no_grad():
                for _ in range(mc_iterations):
                    logits = model(data)
                    probs = torch.softmax(logits, dim=1) # Convert to probabilities
                    mc_predictions.append(probs.numpy())
            
            # Stack all 10 predictions together
            mc_predictions = np.vstack(mc_predictions) # Shape: (10, 5)
            
            # 4. Do the Math (Calculate Mean and Variance)
            mean_probs = np.mean(mc_predictions, axis=0)
            variance = np.var(mc_predictions, axis=0)
            
            # The highest average probability is our final diagnosis
            final_diagnosis = np.argmax(mean_probs)
            max_variance = np.max(variance)
            
            # Create our log message
            log_msg = f"Heartbeat {idx+1} | True: {true_label.item()} | Pred: {final_diagnosis} | Variance: {max_variance:.4f}"
            
            # 5. The Safety Rule (Flag High Uncertainty)
            if max_variance > 0.02:
                log_msg += "  ⚠️ WARNING: HIGH UNCERTAINTY. Flagged for Cardiologist review!"
                uncertain_count += 1
                
            print(log_msg)
            log_file.write(log_msg + "\n")
            
        summary_msg = f"\nEvaluation Complete! Flagged {uncertain_count} out of {num_samples_to_test} heartbeats as UNCERTAIN."
        print(summary_msg)
        log_file.write(summary_msg + "\n")
        
    print(f"\nFull evaluation logs securely registered to: {log_file_path}")

if __name__ == "__main__":
    evaluate_uncertainty()
