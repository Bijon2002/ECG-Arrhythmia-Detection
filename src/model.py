import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

class ConvFeatureExtractor(nn.Module):
    """
    Standardized 3-stage 1D-CNN feature extractor.
    Input: (B, 1, 360)
    Output: (B, 128)
    """
    def __init__(self):
        super(ConvFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 360 -> 180

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 180 -> 90

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 90 -> 45
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 45, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.conv_block(x))

class ECGModelMorphOnly(nn.Module):
    """
    Baseline Morphology-only CNN.
    """
    def __init__(self, num_classes=4):
        super(ECGModelMorphOnly, self).__init__()
        self.extractor = ConvFeatureExtractor()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, rr=None):
        h = self.dropout(self.extractor(x))
        return self.classifier(h)

class ECGModelFusionA(nn.Module):
    """
    Design A: Early/Feature Fusion where RR Branch bypasses dropout.
    Morphology receives Dropout(0.3), RR representation receives 0% dropout.
    """
    def __init__(self, num_classes=4):
        super(ECGModelFusionA, self).__init__()
        self.extractor = ConvFeatureExtractor()
        self.morph_dropout = nn.Dropout(0.3)
        
        # RR projection without dropout
        self.rr_proj = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # Final linear classification layer (no dropout applied to RR)
        self.classifier = nn.Linear(128 + 16, num_classes)

    def forward(self, x, rr):
        h_morph = self.morph_dropout(self.extractor(x))
        h_rr = self.rr_proj(rr)
        h_joint = torch.cat([h_morph, h_rr], dim=1)
        return self.classifier(h_joint)

class ECGModelFusionB(nn.Module):
    """
    Design B: Late Fusion (Logit Sum) with Auxiliary Loss.
    CNN produces z_morph in R^4.
    2-layer RR-MLP produces z_rr in R^4.
    Joint logits = z_morph + z_rr.
    Trained jointly with: L_joint + 0.3*L_morph + 0.3*L_rr.
    """
    def __init__(self, num_classes=4):
        super(ECGModelFusionB, self).__init__()
        self.morph_extractor = ConvFeatureExtractor()
        self.morph_dropout = nn.Dropout(0.3)
        self.morph_head = nn.Linear(128, num_classes)

        # Standalone RR MLP classifier
        self.rr_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, rr, return_aux=False):
        h_morph = self.morph_dropout(self.morph_extractor(x))
        z_morph = self.morph_head(h_morph)
        z_rr = self.rr_head(rr)
        z_joint = z_morph + z_rr

        if return_aux:
            return z_joint, z_morph, z_rr
        return z_joint

if __name__ == "__main__":
    x = torch.randn(8, 1, 360)
    rr = torch.randn(8, 3)

    m0 = ECGModelMorphOnly()
    print("MorphOnly out:", m0(x).shape)

    mA = ECGModelFusionA()
    print("FusionA out:", mA(x, rr).shape)

    mB = ECGModelFusionB()
    z_j, z_m, z_r = mB(x, rr, return_aux=True)
    print("FusionB joint, morph, rr shapes:", z_j.shape, z_m.shape, z_r.shape)
    print("All models verified successfully.")
