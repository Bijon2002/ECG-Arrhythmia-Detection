import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import *
from metrics import clf_metrics, softmax, entropy, cbe, nll, brier, ece, error_detection

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=int, default=0)
ap.add_argument("--weighted", type=int, default=1)
ap.add_argument("--rr", type=int, default=1)
ap.add_argument("--split", default="interpatient", choices=["interpatient", "beatwise"])
ap.add_argument("--mc", type=int, default=10, help="Number of MC Dropout iterations")
ap.add_argument("--flag_q", type=float, default=0.90, help="Quantile for validation flag threshold")
ap.add_argument("--selective", action="store_true", default=True, help="Enable selective prediction (default)")
a = ap.parse_args()

tag = make_tag(a.weighted, a.rr, a.split)
d = load_split(a.split)
dev = device()
n_rr = d["Rtr"].shape[1] if a.rr else 0

models_dir = os.path.join(BASE_DIR, "models", tag)
results_dir = os.path.join(BASE_DIR, "results")
figs_dir = os.path.join(results_dir, "figs")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figs_dir, exist_ok=True)

# 1. Load trained ensemble models
nets = []
for m in range(5):
    pt_path = os.path.join(models_dir, f"run{a.run}_m{m}.pt")
    assert os.path.exists(pt_path), f"Checkpoint not found: {pt_path}"
    net = ECGNet(n_rr=n_rr).to(dev)
    net.load_state_dict(torch.load(pt_path, map_location=dev))
    net.eval()
    nets.append(net)

print(f"Loaded 5 ensemble members for '{tag}' (run {a.run}).")

# 2. Extract logits
lv = [predict_logits(n, d["Xva"], dev, d["Rva"]).numpy() for n in nets]
lt = [predict_logits(n, d["Xte"], dev, d["Rte"]).numpy() for n in nets]

# 3. Fit Temperature Scaling on validation logits
def fit_temperature(logits, y):
    T = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.05, max_iter=50)
    y_t = torch.tensor(y, dtype=torch.long)
    l_t = torch.tensor(logits, dtype=torch.float32)

    def eval_loss():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(l_t / T, y_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return max(float(T.item()), 0.05)

temps = [fit_temperature(lv[i], d["yva"]) for i in range(5)]
print(f"Fitted Temperatures: {[round(t, 3) for t in temps]}")

# 4. Ensemble test evaluation
yte = d["yte"]

# Single baseline model (member 0, uncalibrated)
single_probs = softmax(lt[0])
single = clf_metrics(yte, single_probs.argmax(axis=1))

# Raw ensemble
p_raw = softmax(np.mean(lt, axis=0))
# Calibrated ensemble
lt_cal = np.array([lt[i] / temps[i] for i in range(5)])
p_cal = softmax(np.mean(lt_cal, axis=0))
ens = clf_metrics(yte, p_cal.argmax(axis=1))

cal = {
    "raw": {
        "nll": nll(p_raw, yte),
        "ece": ece(p_raw, yte),
        "brier": brier(p_raw, yte)
    },
    "cal": {
        "nll": nll(p_cal, yte),
        "ece": ece(p_cal, yte),
        "brier": brier(p_cal, yte)
    }
}

print(f"Single Model Accuracy: {single['acc']*100:.2f}% | Macro F1: {single['macro_f1']:.4f}")
print(f"Ensemble Accuracy:     {ens['acc']*100:.2f}% | Macro F1: {ens['macro_f1']:.4f}")
print(f"ECE: Raw={cal['raw']['ece']:.4f} -> Calibrated={cal['cal']['ece']:.4f}")

# 5. MC Dropout Variance
def mc_maxvar(nets, temps, X, R, K, bs=2048):
    if K <= 1:
        return np.zeros(len(X), dtype=np.float32)
    # Enable dropout
    for net in nets:
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    all_passes = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(dev)
            rb = torch.from_numpy(R[i:i + bs]).to(dev) if (R is not None and nets[0].n_rr) else None
            batch_passes = []
            for net, T in zip(nets, temps):
                for _ in range(K // len(nets) + 1):
                    logits = net(xb, rb).float() / T
                    batch_passes.append(torch.softmax(logits, dim=1).cpu().numpy())
            # Stack passes: (passes, batch_size, num_classes)
            bp = np.stack(batch_passes[:K], axis=0)
            # Variance across passes per sample: mean across classes
            b_var = np.mean(np.var(bp, axis=0), axis=1)
            all_passes.append(b_var)

    # Set back to eval mode
    for net in nets:
        net.eval()
    return np.concatenate(all_passes, axis=0)

print(f"Running MC Dropout (K={a.mc} stochastic passes)...")
mcv_val = mc_maxvar(nets, temps, d["Xva"], d["Rva"], a.mc)
mcv_te = mc_maxvar(nets, temps, d["Xte"], d["Rte"], a.mc)

# 6. Uncertainty Metrics & Referral Sweeps
lv_cal = np.array([lv[i] / temps[i] for i in range(5)])
pv_cal = softmax(np.mean(lv_cal, axis=0))

sc_val = {
    "softmax": 1.0 - np.max(pv_cal, axis=1),
    "PE": entropy(pv_cal),
    "CBE": cbe(pv_cal),
    "MCvar": mcv_val
}
sc_te = {
    "softmax": 1.0 - np.max(p_cal, axis=1),
    "PE": entropy(p_cal),
    "CBE": cbe(p_cal),
    "MCvar": mcv_te
}

err = (p_cal.argmax(axis=1) != yte).astype(int)
unc, curves, sweep_mf1 = {}, {}, {}
pred_te = p_cal.argmax(axis=1)

for name in sc_te:
    res, curves[name] = error_detection(sc_te[name], err, refs=(0.05, 0.10, 0.15, 0.20, 0.30))
    thr = float(np.quantile(sc_val[name], a.flag_q))
    flag = sc_te[name] > thr
    res.update(
        thr=thr,
        flag_rate=float(flag.mean()),
        err_caught=float(err[flag].sum() / max(err.sum(), 1)),
        acc_unflagged=float(1.0 - err[~flag].mean()) if (~flag).any() else float("nan")
    )

    order = np.argsort(sc_te[name], kind="stable")
    for r in (0.05, 0.10, 0.15, 0.20, 0.30):
        keep_n = int(round((1 - r) * len(order)))
        res[f"mf1@{int(round(r * 100))}"] = clf_metrics(yte[order[:keep_n]], pred_te[order[:keep_n]])["macro_f1"]

    unc[name] = res

    # Grid for smooth curve
    grid = np.arange(0, 0.31, 0.025)
    sweep_mf1[name] = [
        clf_metrics(yte[order[:int(round((1 - g) * len(order)))]], pred_te[order[:int(round((1 - g) * len(order)))]])["macro_f1"]
        for g in grid
    ]

# 7. Save JSON and NPZ results
out = {
    "tag": tag,
    "run": a.run,
    "temps": temps,
    "n_test": int(len(yte)),
    "err_rate": float(err.mean()),
    "single": single,
    "ensemble": ens,
    "calibration": cal,
    "uncertainty": unc
}

json_path = os.path.join(results_dir, f"{tag}_run{a.run}.json")
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)

scores_path = os.path.join(results_dir, f"{tag}_run{a.run}_scores.npz")
np.savez_compressed(scores_path, p_cal=p_cal, y=yte, **{k: v for k, v in sc_te.items()})

# 8. Generate Publication Figures (IEEE 3.4-inch column width)
# Plot 1: Macro F1 vs Referral Rate
fig, ax = plt.subplots(figsize=(3.4, 2.8))
colors = {"softmax": "#1f77b4", "PE": "#2ca02c", "CBE": "#d62728", "MCvar": "#9467bd"}
for nm, v in sweep_mf1.items():
    ax.plot(grid * 100, v, "o-", ms=2.5, lw=1.2, label=nm, color=colors.get(nm, None))
ax.set_xlabel("Referral rate (% referred to clinician)", fontsize=9)
ax.set_ylabel("Macro F1 on retained beats", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig_f1_path = os.path.join(figs_dir, f"{tag}_run{a.run}_f1_vs_referral.png")
fig.savefig(fig_f1_path, dpi=300)
plt.close(fig)

# Plot 2: Normalized Confusion Matrix
cm = np.array(ens["cm"])
cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(3.4, 3.0))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        color = "white" if cmn[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, fontsize=9)
ax.set_yticklabels(CLASSES, fontsize=9)
ax.set_xlabel("Predicted Diagnosis", fontsize=9)
ax.set_ylabel("Ground Truth", fontsize=9)
fig.tight_layout()
fig_cm_path = os.path.join(figs_dir, f"{tag}_run{a.run}_confusion_matrix.png")
fig.savefig(fig_cm_path, dpi=300)
plt.close(fig)

print(f"\nSaved evaluation metrics to: {json_path}")
print(f"Saved figures to: {fig_f1_path} and {fig_cm_path}")
print(f"Overall Accuracy: {ens['acc']*100:.2f}% | Overall Macro F1: {ens['macro_f1']:.4f}")
