import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import ECGModel

# Bulletproof Absolute Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def load_data():
    print("Loading highly compressed heartbeats...")
    train_data = np.load(os.path.join(PROCESSED_DIR, 'train_data.npz'))
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    
    X_train = torch.tensor(train_data['X'], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_data['y'], dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Loaded {len(train_dataset)} training beats.")
    return train_loader

def train_ensemble(num_models=3, num_epochs=3):
    train_loader = load_data()
    log_file_path = os.path.join(LOGS_DIR, 'ensemble_training_log.txt')
    
    with open(log_file_path, 'w') as log_file:
        log_file.write("ECG Arrhythmia - TRUE DEEP ENSEMBLE Training Log\n")
        log_file.write("================================================\n")
        
        for model_id in range(1, num_models + 1):
            print(f"\n--- Training Model {model_id}/{num_models} ---")
            log_file.write(f"\nTraining Model {model_id}\n")
            
            # Each model gets unique random initialization automatically in PyTorch
            model = ECGModel(num_classes=5)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                
                for batch_idx, (data, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    predictions = model(data)
                    loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                avg_loss = running_loss / len(train_loader)
                log_msg = f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}"
                print(log_msg)
                log_file.write(log_msg + "\n")
                
            model_path = os.path.join(MODELS_DIR, f'ecg_model_v{model_id}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model {model_id} saved to: {model_path}")
            
    print(f"\n🎉 True Deep Ensemble training complete. Logs saved to {log_file_path}")

if __name__ == "__main__":
    # Train 3 completely separate models to form a True Ensemble
    train_ensemble(num_models=3, num_epochs=3)
