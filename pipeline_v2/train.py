import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from common import *
from metrics import clf_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=int, default=0)
ap.add_argument("--weighted", type=int, default=1)
ap.add_argument("--rr", type=int, default=1)
ap.add_argument("--split", default="interpatient", choices=["interpatient", "beatwise"])
ap.add_argument("--members", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()

tag = make_tag(a.weighted, a.rr, a.split)
d = load_split(a.split)
dev = device()

n_rr = d["Rtr"].shape[1] if a.rr else 0
Xtr = torch.from_numpy(d["Xtr"]).unsqueeze(1)
Rtr = torch.from_numpy(d["Rtr"])
ytr = torch.from_numpy(d["ytr"]).long()

counts = np.bincount(d["ytr"], minlength=N_CLASSES)
if a.weighted:
    w = len(ytr) / (N_CLASSES * counts.astype(float) + 1e-5)
    w = w / w.min()  # Normalize minimum weight to 1.0
else:
    w = np.ones(N_CLASSES, dtype=float)

w_tensor = torch.tensor(w, dtype=torch.float32, device=dev)

print("=" * 60)
print(f"Training Tag: {tag} | Device: {dev}")
print(f"Split: {a.split} | Training Beats: {len(Xtr):,} | RR Features: {n_rr}")
print(f"Class Counts: {counts.tolist()} | Weights: {np.round(w, 2).tolist()}")
print("=" * 60)

save_dir = os.path.join(BASE_DIR, "models", tag)
os.makedirs(save_dir, exist_ok=True)

for m in range(a.members):
    seed = 1000 * a.run + m
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ECGNet(n_rr=n_rr).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss(weight=w_tensor)

    best_f1 = -1.0
    best_ep = 0
    bad_epochs = 0
    best_state = None
    hist = []

    print(f"\n>>> Training Member {m+1}/{a.members} (Seed: {seed}) <<<")
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot_loss = 0.0

        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
            rb = Rtr[idx].to(dev) if n_rr else None

            opt.zero_grad()
            out = model(xb, rb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * len(idx)

        # Validation evaluation
        val_logits = predict_logits(model, d["Xva"], dev, d["Rva"]).numpy()
        val_pred = val_logits.argmax(axis=1)
        val_f1 = clf_metrics(d["yva"], val_pred)["macro_f1"]
        mean_loss = tot_loss / len(perm)

        hist.append((ep, mean_loss, val_f1))
        print(f"[{tag} run{a.run} m{m} seed{seed}] Epoch {ep:02d} | Train Loss: {mean_loss:.4f} | Val Macro F1: {val_f1:.4f}", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_ep = ep
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f"Early stopping triggered at epoch {ep} (Best Val Macro F1: {best_f1:.4f} at epoch {best_ep})")
                break

        if ep == a.epochs:
            print(f"Reached max epochs (Best Val Macro F1: {best_f1:.4f} at epoch {best_ep})")

    model_path = os.path.join(save_dir, f"run{a.run}_m{m}.pt")
    torch.save(best_state, model_path)

    meta_path = os.path.join(save_dir, f"run{a.run}_m{m}.json")
    with open(meta_path, "w") as f:
        json.dump(dict(seed=seed, best_epoch=best_ep, best_val_macro_f1=best_f1, history=hist), f, indent=2)

    print(f"Saved member {m} checkpoint to: {model_path}")

print(f"\nAll {a.members} ensemble members successfully trained and saved for tag '{tag}'.")
