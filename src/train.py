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

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    print("Loading highly compressed heartbeats...")
    train_data = np.load(os.path.join(PROCESSED_DIR, 'train_data.npz'))
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    
    # We must format X to have a 'Channels' dimension for PyTorch: (Batch, 1 Channel, 360 Samples)
    X_train = torch.tensor(train_data['X'], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_data['y'], dtype=torch.long)
    
    X_test = torch.tensor(test_data['X'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data['y'], dtype=torch.long)
    
    # Wrap in PyTorch Datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # DataLoaders handle feeding the data in random batches of 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} training beats and {len(test_dataset)} testing beats.")
    return train_loader, test_loader

def train_model(num_epochs=3):
    train_loader, test_loader = load_data()
    
    # 1. Initialize the empty brain
    model = ECGModel(num_classes=5)
    
    # 2. Setup the "Grader" (Loss Function) and "Student" (Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    log_file_path = os.path.join(RESULTS_DIR, 'training_log.txt')
    
    # Open the log file so we can register the results permanently
    with open(log_file_path, 'w') as log_file:
        log_file.write("ECG Arrhythmia Model Training Log\n")
        log_file.write("=================================\n")
        
        print(f"\nTraining Model 1 of the Ensemble for {num_epochs} Epochs...")
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            # --- THE TRAINING LOOP ---
            for batch_idx, (data, labels) in enumerate(train_loader):
                # Erase chalkboard
                optimizer.zero_grad()
                
                # Take the test
                predictions = model(data)
                
                # Grade the test
                loss = criterion(predictions, labels)
                
                # Figure out what went wrong
                loss.backward()
                
                # Study and improve the brain
                optimizer.step()
                
                running_loss += loss.item()
                
            # Calculate the average grade for this epoch
            avg_loss = running_loss / len(train_loader)
            
            log_msg = f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}"
            print(log_msg)
            
            # Register the log to the text file
            log_file.write(log_msg + "\n")
            
    # Save the physical brain to the hard drive
    model_path = os.path.join(MODELS_DIR, 'ecg_model_v1.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"\nModel weights saved perfectly to: {model_path}")
    print(f"Training logs successfully registered to: {log_file_path}")

if __name__ == "__main__":
    # We will train just 1 model for 3 epochs so it is fast to test!
    train_model(num_epochs=3)
