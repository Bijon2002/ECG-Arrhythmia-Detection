import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModelMorphOnly, ECGModelFusionA, ECGModelFusionB

DATA_DIR = os.path.join(BASE_DIR, 'data', 'inter_patient')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['N', 'S', 'V', 'F']

# Rebalanced 5 patient-wise folds across all 22 DS1 records
FOLDS = [
    ['208', '118', '101', '122'],          # Fold 0: F-heavy (208), RBBB-1 (118)
    ['209', '106', '112', '115'],          # Fold 1: S-heavy (209), V-heavy (106)
    ['109', '201', '114', '230'],          # Fold 2: LBBB-1 (109), S-secondary (201)
    ['207', '119', '108', '116'],          # Fold 3: LBBB-2 (207), V-heavy (119)
    ['124', '203', '205', '215', '220', '223'] # Fold 4: RBBB-2 (124), Noise (203)
]

def load_ds1_all():
    train_npz = np.load(os.path.join(DATA_DIR, 'train_data.npz'))
    val_npz = np.load(os.path.join(DATA_DIR, 'val_data.npz'))

    X = np.concatenate([train_npz['X'], val_npz['X']], axis=0)
    RR = np.concatenate([train_npz['RR'], val_npz['RR']], axis=0)
    y = np.concatenate([train_npz['y'], val_npz['y']], axis=0)
    recs = np.concatenate([train_npz['record_ids'], val_npz['record_ids']], axis=0)

    print(f"Total DS1 beats loaded: {len(y):,} across {len(set(recs))} patients.", flush=True)
    return X, RR, y, recs

def get_fold_weights(y_tr, scheme):
    if scheme == 'unweighted':
        return None
    counts = np.bincount(y_tr, minlength=4).astype(float)
    total = len(y_tr)

    if scheme == 'sqrt_inverse':
        w = np.sqrt(total / (counts + 1e-5))
        w = w / w[0]
        w = np.clip(w, 1.0, 10.0)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return torch.tensor(w, dtype=torch.float32)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, digits=4, output_dict=True, zero_division=0)

    f1_n = rep['N']['f1-score']
    f1_s = rep['S']['f1-score']
    f1_v = rep['V']['f1-score']
    macro_f1_nsv = (f1_n + f1_s + f1_v) / 3.0

    n_total = cm[0].sum()
    n_fa = cm[0, 1:].sum()
    fa_rate = n_fa / (n_total + 1e-8)

    return {
        'accuracy': rep['accuracy'],
        'macro_f1_nsv': macro_f1_nsv,
        'macro_f1_all': rep['macro avg']['f1-score'],
        'f1_n': f1_n,
        'f1_s': f1_s,
        'f1_v': f1_v,
        'f1_f': rep['F']['f1-score'],
        's_sens': rep['S']['recall'],
        'v_sens': rep['V']['recall'],
        'f_sens': rep['F']['recall'],
        's_prec': rep['S']['precision'],
        'v_prec': rep['V']['precision'],
        'f_prec': rep['F']['precision'],
        'false_alarm_rate': fa_rate,
        'false_alarm_count': int(n_fa),
        'n_total': int(n_total),
        'confusion_matrix': cm.tolist()
    }

def run_cv_model(arch_type, weight_scheme, X, RR, y, recs, epochs=4, seed=42):
    """
    arch_type: 'morph_only', 'fusion_a', 'fusion_b'
    weight_scheme: 'unweighted', 'sqrt_inverse'
    """
    config_name = f"{arch_type}_{weight_scheme}"
    print(f"\n{'='*75}", flush=True)
    print(f"RUNNING 5-FOLD CV: {config_name} (Epochs={epochs}, Seed={seed})", flush=True)
    print(f"{'='*75}", flush=True)

    oof_preds = np.zeros(len(y), dtype=np.int64)
    oof_preds_zeroed = np.zeros(len(y), dtype=np.int64) if arch_type != 'morph_only' else None
    oof_preds_rr_aux = np.zeros(len(y), dtype=np.int64) if arch_type == 'fusion_b' else None
    fold_results = []

    t0 = time.time()

    for fold_idx, val_records in enumerate(FOLDS):
        val_mask = np.isin(recs, val_records)
        tr_mask = ~val_mask

        # Data tensors
        X_tr = torch.tensor(X[tr_mask], dtype=torch.float32).unsqueeze(1)
        y_tr = torch.tensor(y[tr_mask], dtype=torch.long)
        X_va = torch.tensor(X[val_mask], dtype=torch.float32).unsqueeze(1)
        y_va = torch.tensor(y[val_mask], dtype=torch.long)

        # Standardize RR strictly within training fold
        rr_mean = np.mean(RR[tr_mask], axis=0)
        rr_std = np.std(RR[tr_mask], axis=0) + 1e-6
        RR_tr = torch.tensor((RR[tr_mask] - rr_mean) / rr_std, dtype=torch.float32)
        RR_va = torch.tensor((RR[val_mask] - rr_mean) / rr_std, dtype=torch.float32)

        tr_ds = TensorDataset(X_tr, RR_tr, y_tr)
        va_ds = TensorDataset(X_va, RR_va, y_va)

        tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

        # Weights
        weights = get_fold_weights(y[tr_mask], weight_scheme)
        crit = nn.CrossEntropyLoss(weight=weights) if weights is not None else nn.CrossEntropyLoss()

        torch.manual_seed(seed + fold_idx)
        if arch_type == 'morph_only':
            model = ECGModelMorphOnly()
        elif arch_type == 'fusion_a':
            model = ECGModelFusionA()
        elif arch_type == 'fusion_b':
            model = ECGModelFusionB()
        else:
            raise ValueError(f"Unknown arch: {arch_type}")

        opt = optim.Adam(model.parameters(), lr=0.001)

        # Train for strictly fixed epochs
        for ep in range(1, epochs + 1):
            model.train()
            for xb, rrb, yb in tr_loader:
                opt.zero_grad()
                if arch_type == 'morph_only':
                    logits = model(xb)
                    loss = crit(logits, yb)
                elif arch_type == 'fusion_a':
                    logits = model(xb, rrb)
                    loss = crit(logits, yb)
                elif arch_type == 'fusion_b':
                    z_joint, z_morph, z_rr = model(xb, rrb, return_aux=True)
                    l_j = crit(z_joint, yb)
                    l_m = crit(z_morph, yb)
                    l_r = crit(z_rr, yb)
                    loss = l_j + 0.3 * l_m + 0.3 * l_r
                loss.backward()
                opt.step()

        # Evaluate on validation fold
        model.eval()
        f_preds = []
        f_preds_zeroed = []
        f_preds_rr_aux = []

        with torch.no_grad():
            for xb, rrb, yb in va_loader:
                if arch_type == 'morph_only':
                    out = model(xb)
                    f_preds.extend(torch.argmax(out, dim=1).numpy())
                elif arch_type == 'fusion_a':
                    out = model(xb, rrb)
                    f_preds.extend(torch.argmax(out, dim=1).numpy())
                    # Zeroed RR test
                    out_z = model(xb, torch.zeros_like(rrb))
                    f_preds_zeroed.extend(torch.argmax(out_z, dim=1).numpy())
                elif arch_type == 'fusion_b':
                    z_j, z_m, z_r = model(xb, rrb, return_aux=True)
                    f_preds.extend(torch.argmax(z_j, dim=1).numpy())
                    f_preds_rr_aux.extend(torch.argmax(z_r, dim=1).numpy())
                    # Zeroed RR test
                    out_z = model(xb, torch.zeros_like(rrb))
                    f_preds_zeroed.extend(torch.argmax(out_z, dim=1).numpy())

        f_preds = np.array(f_preds)
        oof_preds[val_mask] = f_preds
        if oof_preds_zeroed is not None:
            oof_preds_zeroed[val_mask] = np.array(f_preds_zeroed)
        if oof_preds_rr_aux is not None:
            oof_preds_rr_aux[val_mask] = np.array(f_preds_rr_aux)

        fm = compute_metrics(y[val_mask], f_preds)
        fold_results.append(fm)

        print(f"  Fold {fold_idx} ({len(val_records)} recs): Macro F1: {fm['macro_f1_nsv']:.4f} | "
              f"S-Sens: {fm['s_sens']*100:.1f}% | V-Sens: {fm['v_sens']*100:.1f}% | "
              f"FAR: {fm['false_alarm_rate']*100:.1f}%", flush=True)

    dur = time.time() - t0
    oof_m = compute_metrics(y, oof_preds)
    mean_macro = float(np.mean([f['macro_f1_nsv'] for f in fold_results]))
    std_macro = float(np.std([f['macro_f1_nsv'] for f in fold_results]))
    mean_far = float(np.mean([f['false_alarm_rate'] for f in fold_results]))

    print(f"\n--- OOF SUMMARY: {config_name} (Duration: {dur:.1f}s) ---", flush=True)
    print(f"  Pooled Macro F1 {{N, S, V}}: {oof_m['macro_f1_nsv']:.4f}")
    print(f"  Mean-of-Folds Macro F1:    {mean_macro:.4f} +/- {std_macro:.4f}")
    print(f"  Accuracy:                  {oof_m['accuracy']*100:.2f}%")
    print(f"  Class S: Sens={oof_m['s_sens']*100:.2f}%, Prec={oof_m['s_prec']*100:.2f}%, F1={oof_m['f1_s']:.4f}")
    print(f"  Class V: Sens={oof_m['v_sens']*100:.2f}%, Prec={oof_m['v_prec']*100:.2f}%, F1={oof_m['f1_v']:.4f}")
    print(f"  Class F: Sens={oof_m['f_sens']*100:.2f}%, Prec={oof_m['f_prec']*100:.2f}%, F1={oof_m['f1_f']:.4f} (Exploratory)")
    print(f"  Class N False Alarm Rate:  {oof_m['false_alarm_rate']*100:.2f}% ({oof_m['false_alarm_count']}/{oof_m['n_total']})")
    print(f"  Mean-of-Folds FAR:         {mean_far*100:.2f}%")

    zeroed_metrics = None
    if oof_preds_zeroed is not None:
        zm = compute_metrics(y, oof_preds_zeroed)
        zeroed_metrics = zm
        print(f"  [Ablation] RR Inputs Zeroed: S-Sens drops {oof_m['s_sens']*100:.1f}% -> {zm['s_sens']*100:.1f}% | Macro F1: {zm['macro_f1_nsv']:.4f}")

    rr_aux_metrics = None
    if oof_preds_rr_aux is not None:
        ram = compute_metrics(y, oof_preds_rr_aux)
        rr_aux_metrics = ram
        print(f"  [Diagnostic] Standalone RR Head (z_RR): S-Sens={ram['s_sens']*100:.1f}% | V-Sens={ram['v_sens']*100:.1f}% | Macro F1={ram['macro_f1_nsv']:.4f}")

    return {
        'config': config_name,
        'arch': arch_type,
        'weight_scheme': weight_scheme,
        'seed': seed,
        'oof_metrics': oof_m,
        'mean_macro_f1': mean_macro,
        'std_macro_f1': std_macro,
        'mean_far': mean_far,
        'fold_results': fold_results,
        'zeroed_metrics': zeroed_metrics,
        'rr_aux_metrics': rr_aux_metrics,
        'oof_preds': oof_preds.tolist()
    }

if __name__ == "__main__":
    X, RR, y, recs = load_ds1_all()

    configs = [
        ('morph_only', 'unweighted'),
        ('morph_only', 'sqrt_inverse'),
        ('fusion_a', 'unweighted'),
        ('fusion_a', 'sqrt_inverse'),
        ('fusion_b', 'unweighted'),
        ('fusion_b', 'sqrt_inverse')
    ]

    all_results = {}
    for arch, weight in configs:
        res = run_cv_model(arch, weight, X, RR, y, recs, epochs=4, seed=42)
        all_results[res['config']] = res

    # Save summary JSON
    out_path = os.path.join(RESULTS_DIR, 'ds1_cv_6configs_rebalanced.json')
    with open(out_path, 'w') as f:
        # Don't save giant pred arrays in small JSON
        summary_to_save = {}
        for k, v in all_results.items():
            summary_to_save[k] = {
                'config': v['config'],
                'arch': v['arch'],
                'weight_scheme': v['weight_scheme'],
                'oof_metrics': v['oof_metrics'],
                'mean_macro_f1': v['mean_macro_f1'],
                'std_macro_f1': v['std_macro_f1'],
                'mean_far': v['mean_far'],
                'fold_results': v['fold_results'],
                'zeroed_metrics': v['zeroed_metrics'],
                'rr_aux_metrics': v['rr_aux_metrics']
            }
        json.dump(summary_to_save, f, indent=2)
    print(f"\nAll 6 configs complete. Results saved to {out_path}", flush=True)
