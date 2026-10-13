import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import numpy as np
import scipy.signal as sg
import wfdb
from sklearn.model_selection import train_test_split
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="interpatient", choices=["interpatient", "beatwise"])
ap.add_argument("--no-paced", action="store_true")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

RAW = os.path.join(DATA_DIR, "raw")
OUT = DATA_DIR
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

sos = sg.butter(2, [0.5, 45], btype="band", fs=360, output="sos")

def rr_features(samples):
    """
    Extracts 4 RR-interval timing features per beat:
    1. Previous RR
    2. Next RR
    3. Local mean RR (window of 10 previous beats)
    4. Ratio: previous RR / local mean RR
    """
    n = len(samples)
    if n < 2:
        return np.zeros((n, 4), dtype=np.float32)
    rr = np.diff(samples) / 360.0
    prev = np.concatenate([[rr[0]], rr])
    nxt = np.concatenate([rr, [rr[-1]]])
    c = np.cumsum(np.concatenate([[0.0], prev]))
    idx = np.arange(n)
    lo = np.maximum(0, idx - 9)
    local = (c[idx + 1] - c[lo]) / (idx + 1 - lo)
    ratio = prev / np.maximum(local, 1e-3)

    f = np.stack([prev, nxt, local, ratio], 1)
    f[:, :3] = np.clip(f[:, :3], 0.15, 3.0)
    f[:, 3] = np.clip(f[:, 3], 0.2, 5.0)
    return f.astype(np.float32)

def load_record(rec):
    rec_path = os.path.join(RAW, rec)
    r = wfdb.rdrecord(rec_path)
    ch = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    sig = sg.sosfiltfilt(sos, r.p_signal[:, ch])
    ann = wfdb.rdann(rec_path, "atr")

    keep = [(s, AAMI[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
    if not keep:
        return np.empty((0, 360), dtype=np.float32), np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32)

    samples = np.array([k[0] for k in keep])
    labels = np.array([k[1] for k in keep])
    feats = rr_features(samples)

    X, y, R = [], [], []
    for s, lab, f in zip(samples, labels, feats):
        if s - 180 < 0 or s + 180 > len(sig):
            continue
        w = sig[s - 180:s + 180]
        w = (w - w.mean()) / (w.std() + 1e-8)  # per-beat z-score
        X.append(w.astype(np.float32))
        y.append(lab)
        R.append(f)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(R, dtype=np.float32)

if args.mode == "interpatient":
    train_recs = [r for r in DS1 if r not in VAL] + ([] if args.no_paced else PACED_TRAIN)
    val_recs = list(VAL)
    test_recs = DS2 + ([] if args.no_paced else PACED_TEST)
    recs = train_recs + val_recs + test_recs
else:
    recs = DS1 + DS2 + ([] if args.no_paced else PACED_TRAIN + PACED_TEST)

print(f"Loading {len(recs)} records (mode={args.mode}, no_paced={args.no_paced})...")
data = {}
for rec in recs:
    data[rec] = load_record(rec)
    print(f"Record {rec}: {len(data[rec][0]):,} beats, counts: {np.bincount(data[rec][1], minlength=N_CLASSES).tolist()}")

def stack(rs):
    valid_rs = [r for r in rs if len(data[r][0]) > 0]
    X = np.concatenate([data[r][0] for r in valid_rs], axis=0)
    y = np.concatenate([data[r][1] for r in valid_rs], axis=0)
    R = np.concatenate([data[r][2] for r in valid_rs], axis=0)
    rid = np.concatenate([[r] * len(data[r][1]) for r in valid_rs], axis=0)
    return X, y, R, rid

if args.mode == "interpatient":
    Xtr, ytr, Rtr, rtr = stack(train_recs)
    Xva, yva, Rva, rva = stack(val_recs)
    Xte, yte, Rte, rte = stack(test_recs)
    fname, cname = "split.npz", "split_counts.json"
    train_info, val_info, test_info = train_recs, val_recs, test_recs
else:
    X, y, R, rid = stack(recs)
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=args.seed)
    i_tr, i_va = train_test_split(i_tr, test_size=0.1, stratify=y[i_tr], random_state=args.seed)
    Xtr, ytr, Rtr, rtr = X[i_tr], y[i_tr], R[i_tr], rid[i_tr]
    Xva, yva, Rva, rva = X[i_va], y[i_va], R[i_va], rid[i_va]
    Xte, yte, Rte, rte = X[i_te], y[i_te], R[i_te], rid[i_te]
    fname, cname = "split_beatwise.npz", "split_counts_beatwise.json"
    train_info = val_info = test_info = "random beats from all records"

# Standardize RR features with train statistics
mu = Rtr.mean(axis=0)
sd = Rtr.std(axis=0) + 1e-6
Rtr = (Rtr - mu) / sd
Rva = (Rva - mu) / sd
Rte = (Rte - mu) / sd

out_path = os.path.join(OUT, fname)
np.savez_compressed(
    out_path,
    Xtr=Xtr, ytr=ytr, rtr=rtr, Rtr=Rtr,
    Xva=Xva, yva=yva, rva=rva, Rva=Rva,
    Xte=Xte, yte=yte, rte=rte, Rte=Rte
)
print(f"\nSaved split file: {out_path}")

counts = {
    name: np.bincount(yy, minlength=N_CLASSES).tolist()
    for name, yy in [("train", ytr), ("val", yva), ("test", yte)]
}
counts["records"] = {"train": train_info, "val": val_info, "test": test_info}

json_path = os.path.join(OUT, cname)
with open(json_path, "w") as f:
    json.dump(counts, f, indent=2)

print(f"\nMode={args.mode} Class counts (N, S, V, F, Q):")
for k in ("train", "val", "test"):
    print(f"  {k:6s}: {counts[k]} | Total: {sum(counts[k]):,}")
