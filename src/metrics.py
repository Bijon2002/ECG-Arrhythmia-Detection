import os
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to our 'src' folder
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModel

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Define our correct subfolders
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 0. Clean up our lazy shortcut from yesterday (Move files to /logs/)
for log_file in ['training_log.txt', 'evaluation_log.txt']:
    old_path = os.path.join(RESULTS_DIR, log_file)
    new_path = os.path.join(LOGS_DIR, log_file)
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        print(f"Moved {log_file} into the correct /logs/ folder!")

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]

def generate_metrics():
    print("\nLoading 21,888 unseen testing heartbeats...")
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    
    X_test = torch.tensor(test_data['X'], dtype=torch.float32).unsqueeze(1)
    y_true = torch.tensor(test_data['y'], dtype=torch.long)
    
    # We can use a massive batch size (256) since we are just testing, not training
    test_dataset = TensorDataset(X_test, y_true)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print("Loading Trained AI Brain...")
    model = ECGModel(num_classes=5)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ecg_model_v1.pth'), weights_only=True))
    model.eval() # Turn off Dropout completely for pure accuracy testing
    
    print("Diagnosing all 21,888 heartbeats...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            logits = model(data)
            preds = torch.argmax(logits, dim=1) # Get the most confident diagnosis
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 1. Calculate Formal F1 Scores using scikit-learn
    print("\nCalculating Final Grades (F1 Scores)...")
    
    # The zero_division=0 prevents warnings if a class has 0 predictions
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)
    
    report_path = os.path.join(METRICS_DIR, 'f1_scores.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ECG Arrhythmia AI - Final Evaluation Metrics\n")
        f.write("============================================\n\n")
        f.write(report)
        
    print(report)
    print(f"✅ F1 Scores saved securely to: {report_path}")
    
    # 2. Draw the Confusion Matrix Image
    print("\nDrawing Confusion Matrix Image...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('AI Confusion Matrix (Where does it get confused?)', fontsize=14)
    plt.ylabel('True Diagnosis (Ground Truth)', fontsize=12)
    plt.xlabel('AI Diagnosis (Prediction)', fontsize=12)
    
    plot_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Confusion Matrix Image saved to: {plot_path}")

if __name__ == "__main__":
    generate_metrics()
