"""
compute_ece.py — Expected Calibration Error (ECE) calculation
Computes ECE and Maximum Calibration Error (MCE) for the ensemble
BEFORE and AFTER temperature scaling, producing a reliability diagram
and saving all results to results/metrics/calibration_results.txt

Run from the project root:
    python src/compute_ece.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModel

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
METRICS_DIR   = os.path.join(BASE_DIR, 'results', 'metrics')
PLOTS_DIR     = os.path.join(BASE_DIR, 'results', 'plots')

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

NUM_BINS = 10  # Standard for ECE (Guo et al., 2017)


def load_ensemble(with_temperature: bool):
    """Load all 5 ensemble members. Apply temperature if requested."""
    models, temps = [], []
    for i in range(1, 6):
        m = ECGModel(num_classes=5)
        m.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, f'ecg_model_v{i}.pth'),
                       map_location='cpu', weights_only=True)
        )
        m.eval()
        models.append(m)

        t_path = os.path.join(MODELS_DIR, f'temperature_v{i}.txt')
        t = float(open(t_path).read().strip()) if (with_temperature and os.path.exists(t_path)) else 1.0
        temps.append(t)
    return models, temps


def get_ensemble_confidences(models, temps, loader):
    """
    Run the full test set through the ensemble.
    Returns:
        confidences: np.ndarray shape (N,)   — max predicted probability
        accuracies:  np.ndarray shape (N,)   — 1 if correct, 0 if not
    """
    all_confidences, all_correct = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            # Collect per-model calibrated probabilities
            batch_probs = []
            for model, temp in zip(models, temps):
                logits = model(X_batch)
                probs  = torch.softmax(logits / temp, dim=1).numpy()
                batch_probs.append(probs)

            mean_probs    = np.mean(batch_probs, axis=0)          # (B, 5)
            pred_classes  = np.argmax(mean_probs, axis=1)         # (B,)
            max_conf      = np.max(mean_probs, axis=1)            # (B,)
            correct       = (pred_classes == y_batch.numpy()).astype(float)

            all_confidences.extend(max_conf.tolist())
            all_correct.extend(correct.tolist())

    return np.array(all_confidences), np.array(all_correct)


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = NUM_BINS):
    """
    Expected Calibration Error — equal-width binning.
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bins       = np.linspace(0.0, 1.0, n_bins + 1)
    ece        = 0.0
    mce        = 0.0
    N          = len(confidences)
    bin_data   = []   # for the reliability diagram

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        # Include upper boundary only for the last bin
        mask = (confidences > lo) & (confidences <= hi) if b < n_bins - 1 \
               else (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            bin_data.append((0.5 * (lo + hi), 0.0, 0.0, 0))
            continue

        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_n    = mask.sum()

        gap  = abs(bin_acc - bin_conf)
        ece += (bin_n / N) * gap
        mce  = max(mce, gap)
        bin_data.append((bin_conf, bin_acc, gap, int(bin_n)))

    return ece, mce, bin_data


def plot_reliability_diagram(bin_data_before, bin_data_after,
                             ece_before, ece_after, save_path):
    """Reliability diagram: confidence vs accuracy, before and after calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bin_data, ece_val, title in zip(
            axes,
            [bin_data_before, bin_data_after],
            [ece_before, ece_after],
            ['Before Temperature Scaling', 'After Temperature Scaling']):

        confs = [b[0] for b in bin_data]
        accs  = [b[1] for b in bin_data]

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
        ax.bar(confs, accs, width=0.08, alpha=0.7,
               color='steelblue', edgecolor='navy', label='Ensemble')
        ax.bar(confs, confs, width=0.08, alpha=0.3,
               color='red', edgecolor='darkred', label='Gap (overconfidence)')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Confidence', fontsize=12)
        ax.set_ylabel('Fraction of Correct Predictions', fontsize=12)
        ax.set_title(f'{title}\nECE = {ece_val:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagram saved to: {save_path}")


def main():
    print("=" * 60)
    print("ECG Ensemble — Expected Calibration Error (ECE) Analysis")
    print("=" * 60)

    # Load test set (use full 21,888 beats for ECE — not just calibration subset)
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    X_test = torch.tensor(test_data['X'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_data['y'], dtype=torch.long)
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)

    print(f"\nTest set: {len(X_test):,} heartbeats")
    print(f"Calibration bins: {NUM_BINS}")

    # ── BEFORE temperature scaling (T=1.0 for all models) ──────────────────
    print("\n[1/2] Computing ECE BEFORE temperature scaling (T=1.0)...")
    models_raw, temps_raw = load_ensemble(with_temperature=False)
    conf_before, acc_before = get_ensemble_confidences(models_raw, temps_raw, loader)
    ece_before, mce_before, bins_before = compute_ece(conf_before, acc_before)

    # ── AFTER temperature scaling ───────────────────────────────────────────
    print("[2/2] Computing ECE AFTER temperature scaling...")
    models_cal, temps_cal = load_ensemble(with_temperature=True)
    conf_after, acc_after = get_ensemble_confidences(models_cal, temps_cal, loader)
    ece_after, mce_after, bins_after = compute_ece(conf_after, acc_after)

    # ── Print results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Before T-Scaling':>18} {'After T-Scaling':>16}")
    print("-" * 66)
    print(f"{'ECE (↓ better)':<30} {ece_before:>18.4f} {ece_after:>16.4f}")
    print(f"{'MCE (↓ better)':<30} {mce_before:>18.4f} {mce_after:>16.4f}")
    reduction = 100 * (ece_before - ece_after) / (ece_before + 1e-10)
    print(f"\nECE reduction: {reduction:.1f}%")

    # ── Save text report ────────────────────────────────────────────────────
    report_path = os.path.join(METRICS_DIR, 'calibration_results.txt')
    with open(report_path, 'w') as f:
        f.write("ECG Ensemble — Calibration Analysis (ECE / MCE)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test set size:          {len(X_test):,} heartbeats\n")
        f.write(f"Number of ECE bins:     {NUM_BINS}\n\n")
        f.write(f"ECE BEFORE T-Scaling:   {ece_before:.4f}\n")
        f.write(f"MCE BEFORE T-Scaling:   {mce_before:.4f}\n\n")
        f.write(f"ECE AFTER  T-Scaling:   {ece_after:.4f}\n")
        f.write(f"MCE AFTER  T-Scaling:   {mce_after:.4f}\n\n")
        f.write(f"ECE Reduction:          {reduction:.1f}%\n\n")
        f.write("Temperature values used:\n")
        for i, t in enumerate(temps_cal, 1):
            f.write(f"  Model {i}: T = {t:.4f}\n")
    print(f"\nText report saved to:  {report_path}")

    # ── Save reliability diagram ────────────────────────────────────────────
    diagram_path = os.path.join(PLOTS_DIR, 'reliability_diagram.png')
    plot_reliability_diagram(bins_before, bins_after,
                             ece_before, ece_after, diagram_path)

    print("\n✅ ECE analysis complete.")
    return ece_before, ece_after, reduction


if __name__ == '__main__':
    main()
