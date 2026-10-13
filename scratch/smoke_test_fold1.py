import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = r"f:\ECG-Arrhythmia-Detection"
DATA_DIR = os.path.join(BASE_DIR, 'data', 'inter_patient')
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModelMorphOnly, ECGModelFusionA, ECGModelFusionB

train_npz = np.load(os.path.join(DATA_DIR, 'train_data.npz'))
val_npz = np.load(os.path.join(DATA_DIR, 'val_data.npz'))

X = np.concatenate([train_npz['X'], val_npz['X']], axis=0)
RR = np.concatenate([train_npz['RR'], val_npz['RR']], axis=0)
y = np.concatenate([train_npz['y'], val_npz['y']], axis=0)
recs = np.concatenate([train_npz['record_ids'], val_npz['record_ids']], axis=0)

CLASS_NAMES = ['N', 'S', 'V', 'F']
FOLD1_RECS = ['209', '106', '112', '115']

val_mask = np.isin(recs, FOLD1_RECS)
tr_mask = ~val_mask

# Waveforms
X_tr = torch.tensor(X[tr_mask], dtype=torch.float32).unsqueeze(1)
y_tr = torch.tensor(y[tr_mask], dtype=torch.long)
X_va = torch.tensor(X[val_mask], dtype=torch.float32).unsqueeze(1)
y_va = torch.tensor(y[val_mask], dtype=torch.long)

# Standardize RR features strictly within training fold
rr_mean = np.mean(RR[tr_mask], axis=0)
rr_std = np.std(RR[tr_mask], axis=0) + 1e-6

RR_tr = torch.tensor((RR[tr_mask] - rr_mean) / rr_std, dtype=torch.float32)
RR_va = torch.tensor((RR[val_mask] - rr_mean) / rr_std, dtype=torch.float32)

tr_ds = TensorDataset(X_tr, RR_tr, y_tr)
va_ds = TensorDataset(X_va, RR_va, y_va)

tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

crit = nn.CrossEntropyLoss()

def eval_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    m_f1 = (rep['N']['f1-score'] + rep['S']['f1-score'] + rep['V']['f1-score']) / 3.0
    far = cm[0, 1:].sum() / (cm[0].sum() + 1e-8)
    return {
        'macro_f1_nsv': m_f1,
        's_sens': rep['S']['recall'],
        'v_sens': rep['V']['recall'],
        'f_sens': rep['F']['recall'],
        'far': far,
        'cm': cm
    }

print("=" * 80)
print("ONE-FOLD SMOKE TEST ON FOLD 1 (Holds Record 209: 383 S beats)")
print("Fixed 4 Epochs | Unweighted Loss")
print("=" * 80)

# --- 1. Baseline Morph Only ---
print("\n1. Training Baseline Morphology-only CNN (4 epochs)...", flush=True)
torch.manual_seed(42)
m0 = ECGModelMorphOnly()
opt0 = optim.Adam(m0.parameters(), lr=0.001)

for epoch in range(1, 5):
    m0.train()
    ep_loss = 0.0
    for xb, rrb, yb in tr_loader:
        opt0.zero_grad()
        out = m0(xb)
        loss = crit(out, yb)
        loss.backward()
        opt0.step()
        ep_loss += loss.item() * len(yb)
    print(f"  Epoch {epoch}/4 - Loss: {ep_loss/len(y_tr):.4f}", flush=True)

m0.eval()
preds_m0 = []
with torch.no_grad():
    for xb, rrb, yb in va_loader:
        preds_m0.extend(torch.argmax(m0(xb), dim=1).numpy())
res_m0 = eval_metrics(y[val_mask], np.array(preds_m0))
print(f"  MorphOnly: Macro F1={res_m0['macro_f1_nsv']:.4f} | S-Sens={res_m0['s_sens']*100:.2f}% | V-Sens={res_m0['v_sens']*100:.2f}% | FAR={res_m0['far']*100:.2f}%")

# --- 2. Design A: Dropout Bypass ---
print("\n2. Training Design A (RR Bypasses Dropout, 4 epochs)...", flush=True)
torch.manual_seed(42)
mA = ECGModelFusionA()
optA = optim.Adam(mA.parameters(), lr=0.001)

for epoch in range(1, 5):
    mA.train()
    ep_loss = 0.0
    for xb, rrb, yb in tr_loader:
        optA.zero_grad()
        out = mA(xb, rrb)
        loss = crit(out, yb)
        loss.backward()
        optA.step()
        ep_loss += loss.item() * len(yb)
    print(f"  Epoch {epoch}/4 - Loss: {ep_loss/len(y_tr):.4f}", flush=True)

mA.eval()
preds_mA, preds_mA_zeroed = [], []
with torch.no_grad():
    for xb, rrb, yb in va_loader:
        # Normal inference
        preds_mA.extend(torch.argmax(mA(xb, rrb), dim=1).numpy())
        # Ablation: RR inputs zeroed out
        preds_mA_zeroed.extend(torch.argmax(mA(xb, torch.zeros_like(rrb)), dim=1).numpy())

res_mA = eval_metrics(y[val_mask], np.array(preds_mA))
res_mA_zeroed = eval_metrics(y[val_mask], np.array(preds_mA_zeroed))

print(f"  Design A Normal:   Macro F1={res_mA['macro_f1_nsv']:.4f} | S-Sens={res_mA['s_sens']*100:.2f}% | V-Sens={res_mA['v_sens']*100:.2f}% | FAR={res_mA['far']*100:.2f}%")
print(f"  Design A (RR=0):   Macro F1={res_mA_zeroed['macro_f1_nsv']:.4f} | S-Sens={res_mA_zeroed['s_sens']*100:.2f}% | V-Sens={res_mA_zeroed['v_sens']*100:.2f}% | FAR={res_mA_zeroed['far']*100:.2f}%")
print(f"  --> S-Sensitivity Drop under RR Zeroing: {res_mA['s_sens']*100:.1f}% -> {res_mA_zeroed['s_sens']*100:.1f}%")

# --- 3. Design B: Late Fusion with Aux Loss ---
print("\n3. Training Design B (Late Fusion with Auxiliary Loss, 4 epochs)...", flush=True)
torch.manual_seed(42)
mB = ECGModelFusionB()
optB = optim.Adam(mB.parameters(), lr=0.001)

for epoch in range(1, 5):
    mB.train()
    ep_loss = 0.0
    for xb, rrb, yb in tr_loader:
        optB.zero_grad()
        z_joint, z_morph, z_rr = mB(xb, rrb, return_aux=True)
        l_j = crit(z_joint, yb)
        l_m = crit(z_morph, yb)
        l_r = crit(z_rr, yb)
        loss = l_j + 0.3 * l_m + 0.3 * l_r
        loss.backward()
        optB.step()
        ep_loss += loss.item() * len(yb)
    print(f"  Epoch {epoch}/4 - Loss: {ep_loss/len(y_tr):.4f}", flush=True)

mB.eval()
preds_mB, preds_mB_rr_only, preds_mB_zeroed = [], [], []
with torch.no_grad():
    for xb, rrb, yb in va_loader:
        z_joint, z_morph, z_rr = mB(xb, rrb, return_aux=True)
        preds_mB.extend(torch.argmax(z_joint, dim=1).numpy())
        preds_mB_rr_only.extend(torch.argmax(z_rr, dim=1).numpy())
        # Ablation: zero RR
        preds_mB_zeroed.extend(torch.argmax(mB(xb, torch.zeros_like(rrb)), dim=1).numpy())

res_mB = eval_metrics(y[val_mask], np.array(preds_mB))
res_mB_rr_only = eval_metrics(y[val_mask], np.array(preds_mB_rr_only))
res_mB_zeroed = eval_metrics(y[val_mask], np.array(preds_mB_zeroed))

print(f"  Design B Joint:     Macro F1={res_mB['macro_f1_nsv']:.4f} | S-Sens={res_mB['s_sens']*100:.2f}% | V-Sens={res_mB['v_sens']*100:.2f}% | FAR={res_mB['far']*100:.2f}%")
print(f"  Design B (z_RR):    Macro F1={res_mB_rr_only['macro_f1_nsv']:.4f} | S-Sens={res_mB_rr_only['s_sens']*100:.2f}% | V-Sens={res_mB_rr_only['v_sens']*100:.2f}% | FAR={res_mB_rr_only['far']*100:.2f}%")
print(f"  Design B (RR=0):    Macro F1={res_mB_zeroed['macro_f1_nsv']:.4f} | S-Sens={res_mB_zeroed['s_sens']*100:.2f}% | V-Sens={res_mB_zeroed['v_sens']*100:.2f}% | FAR={res_mB_zeroed['far']*100:.2f}%")
print(f"  --> Standalone RR Head S-Sensitivity: {res_mB_rr_only['s_sens']*100:.1f}%")
print(f"  --> S-Sensitivity Drop under RR Zeroing: {res_mB['s_sens']*100:.1f}% -> {res_mB_zeroed['s_sens']*100:.1f}%")
