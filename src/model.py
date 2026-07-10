import torch
import torch.nn as nn

class ECGModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGModel, self).__init__()
        
        # --- 1D Convolutional Blocks (The Scanners) ---
        # Input shape expected: (Batch Size, 1 Channel, 360 Samples)
        
        # Block 1: Basic Feature Detection
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Shrinks from 360 -> 180 samples
        )
        
        # Block 2: Complex Feature Detection
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Shrinks from 180 -> 90 samples
        )
        
        # Block 3: Deep Feature Detection
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Shrinks from 90 -> 45 samples
        )
        
        # --- Fully Connected Classifier (The Judge / Decision Room) ---
        # After scanning, we have 128 channels of data, each 45 samples long. 
        # 128 * 45 = 5,760 mathematical clues!
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 45, 256),  # Connect clues to 256 neurons
            nn.ReLU(),
            nn.Dropout(0.5),           # Randomly drop 50% to prevent overfitting/memorization
            nn.Linear(256, num_classes) # Final 5 outputs for our 5 AAMI classes
        )

    def forward(self, x):
        """
        This defines how the heartbeat travels through the brain.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # --- Automated Test ---
    print("Testing ECGModel architecture...")
    
    # 1. Create a fake heartbeat (Batch=1, Channels=1, Samples=360)
    fake_heartbeat = torch.randn(1, 1, 360)
    
    # 2. Initialize our AI brain
    model = ECGModel(num_classes=5)
    
    # 3. Ask the brain to diagnose the fake heartbeat
    predictions = model(fake_heartbeat)
    
    print(f"Input shape: {fake_heartbeat.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # 4. Verify it spit out exactly 5 numbers
    if predictions.shape == (1, 5):
        print("✅ SUCCESS! The model successfully scanned 360 samples and output 5 class probabilities.")
    else:
        print("❌ ERROR: The output shape is incorrect.")
