import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLASSES = ['N', 'S', 'V', 'F', 'Q']
N_CLASSES = 5

AAMI = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

DS1 = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]
VAL = {'114', '124', '205', '223'}
DS2 = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]
PACED_TRAIN = ['102', '104']
PACED_TEST = ['107', '217']

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ECGNet(nn.Module):
    """3 conv blocks + FC classifier with optional RR-interval features."""
    def __init__(self, n_classes=N_CLASSES, p=0.5, n_rr=0):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv1d(i, o, 5, padding=2),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        self.n_rr = n_rr
        self.features = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 45 + n_rr, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, rr=None):
        h = self.flat(self.features(x))
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], 1)
        return self.fc(h)

def make_tag(weighted, rr, split):
    return ("weighted" if weighted else "unweighted") + \
           ("_rr" if rr else "") + \
           ("_bw" if split == "beatwise" else "")

def load_split(split="interpatient"):
    filename = "split.npz" if split == "interpatient" else "split_beatwise.npz"
    path = os.path.join(DATA_DIR, filename)
    d = np.load(path)
    return {k: d[k] for k in d.files}

@torch.no_grad()
def predict_logits(model, X, dev, R=None, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(dev)
        rb = torch.from_numpy(R[i:i + bs]).to(dev) if (R is not None and model.n_rr) else None
        out.append(model(xb, rb).float().cpu())
    return torch.cat(out)
