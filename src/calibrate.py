import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import ECGModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# The temperature parameter is a single scalar optimized to improve confidence calibration
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Start at T=1.5

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def calibrate_ensemble():
    print("Loading test data for Temperature Scaling Calibration...")
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    X_test = torch.tensor(test_data['X'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data['y'], dtype=torch.long)
    
    # We calibrate using a smaller subset to avoid overfitting
    calib_dataset = TensorDataset(X_test[:2000], y_test[:2000])
    calib_loader = DataLoader(calib_dataset, batch_size=256, shuffle=False)
    
    nll_criterion = nn.CrossEntropyLoss()
    
    for i in range(1, 4):
        print(f"\nCalibrating Model {i}...")
        model = ECGModel(num_classes=5)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'ecg_model_v{i}.pth'), weights_only=True))
        model.eval()
        
        scaled_model = ModelWithTemperature(model)
        optimizer = optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = 0.0
            for data, labels in calib_loader:
                out = scaled_model(data)
                loss += nll_criterion(out, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        
        optimal_t = scaled_model.temperature.item()
        print(f"Optimal Temperature (T) for Model {i}: {optimal_t:.4f}")
        
        # Save the temperature scalar to a text file so our API can use it
        with open(os.path.join(MODELS_DIR, f'temperature_v{i}.txt'), 'w') as f:
            f.write(str(optimal_t))
            
    print("\n✅ All ensemble models successfully calibrated.")

if __name__ == "__main__":
    calibrate_ensemble()
