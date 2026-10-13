import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = r"f:\ECG-Arrhythmia-Detection"
DATA_DIR = os.path.join(BASE_DIR, 'data', 'inter_patient')

train_npz = np.load(os.path.join(DATA_DIR, 'train_data.npz'))
val_npz = np.load(os.path.join(DATA_DIR, 'val_data.npz'))

X = np.concatenate([train_npz['X'], val_npz['X']], axis=0)
RR = np.concatenate([train_npz['RR'], val_npz['RR']], axis=0)
y = np.concatenate([train_npz['y'], val_npz['y']], axis=0)
recs = np.concatenate([train_npz['record_ids'], val_npz['record_ids']], axis=0)

FOLDS = [
    ['208', '118', '101', '122'],          # Fold 0
    ['209', '106', '112', '115'],          # Fold 1
    ['109', '201', '114', '230'],          # Fold 2
    ['207', '119', '108', '116'],          # Fold 3
    ['124', '203', '205', '215', '220', '223'] # Fold 4
]

CLASS_NAMES = ['N', 'S', 'V', 'F']

print("================================================================================")
print("PART 1: EXACT SCRIPT-GENERATED FOLD TABLE (Directly from record_ids & y)")
print("================================================================================")

# First compute per-record counts
rec_counts = {}
for r in sorted(list(set(recs))):
    mask = (recs == r)
    y_r = y[mask]
    counts = [int(np.sum(y_r == c)) for c in range(4)]
    rec_counts[r] = counts

fold_stats = []
total_all = [0, 0, 0, 0, 0]

print(f"{'Fold':<8} {'Records':<42} {'N':<8} {'S':<6} {'V':<6} {'F':<6} {'Total':<8}")
print("-" * 80)

for f_idx, fold_recs in enumerate(FOLDS):
    f_counts = np.zeros(4, dtype=int)
    for r in fold_recs:
        f_counts += rec_counts[r]
    f_tot = int(np.sum(f_counts))
    recs_str = str(fold_recs)
    print(f"Fold {f_idx:<3} {recs_str:<42} {f_counts[0]:<8} {f_counts[1]:<6} {f_counts[2]:<6} {f_counts[3]:<6} {f_tot:<8}")
    for c in range(4):
        total_all[c] += f_counts[c]
    total_all[4] += f_tot

print("-" * 80)
print(f"{'TOTAL':<8} {'All 22 DS1 Records':<42} {total_all[0]:<8} {total_all[1]:<6} {total_all[2]:<6} {total_all[3]:<6} {total_all[4]:<8}")

print("\n\nPER-RECORD BREAKDOWN:")
print(f"{'Record':<8} {'N':<8} {'S':<6} {'V':<6} {'F':<6} {'Total':<8}")
print("-" * 40)
for r in sorted(list(rec_counts.keys())):
    c = rec_counts[r]
    print(f"{r:<8} {c[0]:<8} {c[1]:<6} {c[2]:<6} {c[3]:<6} {sum(c):<8}")

print("\n" + "=" * 80)
print("PART 2: RECOMPUTE RR-ONLY LOGISTIC REGRESSION ON REBALANCED FOLDS")
print("================================================================================")

oof_preds_rr = np.zeros(len(y), dtype=int)
fold_metrics_rr = []

for f_idx, fold_recs in enumerate(FOLDS):
    val_mask = np.isin(recs, fold_recs)
    tr_mask = ~val_mask
    
    # Standardize RR features strictly within fold
    rr_mean = np.mean(RR[tr_mask], axis=0)
    rr_std = np.std(RR[tr_mask], axis=0) + 1e-6
    
    RR_tr = (RR[tr_mask] - rr_mean) / rr_std
    y_tr = y[tr_mask]
    
    RR_va = (RR[val_mask] - rr_mean) / rr_std
    y_va = y[val_mask]
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42 + f_idx)
    clf.fit(RR_tr, y_tr)
    
    preds = clf.predict(RR_va)
    oof_preds_rr[val_mask] = preds
    
    cm = confusion_matrix(y_va, preds, labels=[0, 1, 2, 3])
    rep = classification_report(y_va, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    n_tot = cm[0].sum()
    fa = cm[0, 1:].sum()
    far = fa / (n_tot + 1e-8)
    
    m_f1 = (rep['N']['f1-score'] + rep['S']['f1-score'] + rep['V']['f1-score']) / 3.0
    s_rec = rep['S']['recall']
    v_rec = rep['V']['recall']
    
    fold_metrics_rr.append({
        'fold': f_idx,
        'macro_f1_nsv': m_f1,
        'f1_n': rep['N']['f1-score'],
        'f1_s': rep['S']['f1-score'],
        'f1_v': rep['V']['f1-score'],
        's_sens': s_rec,
        'v_sens': v_rec,
        'far': far,
        'cm': cm
    })
    
    print(f"Fold {f_idx}: Macro F1 (N/S/V) = {m_f1:.4f} | S-Sens = {s_rec*100:.2f}% | V-Sens = {v_rec*100:.2f}% | FAR = {far*100:.2f}%")

print("-" * 80)
# Pooled OOF
rep_oof = classification_report(y, oof_preds_rr, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
cm_oof = confusion_matrix(y, oof_preds_rr, labels=[0, 1, 2, 3])
fa_oof = cm_oof[0, 1:].sum()
n_tot_oof = cm_oof[0].sum()
far_oof = fa_oof / (n_tot_oof + 1e-8)

macro_f1_nsv_oof = (rep_oof['N']['f1-score'] + rep_oof['S']['f1-score'] + rep_oof['V']['f1-score']) / 3.0
mean_of_folds_f1 = np.mean([fm['macro_f1_nsv'] for fm in fold_metrics_rr])
std_of_folds_f1 = np.std([fm['macro_f1_nsv'] for fm in fold_metrics_rr])

mean_s_sens = np.mean([fm['s_sens'] for fm in fold_metrics_rr])
mean_v_sens = np.mean([fm['v_sens'] for fm in fold_metrics_rr])
mean_far = np.mean([fm['far'] for fm in fold_metrics_rr])

print(f"\nPOOLED OOF RESULTS (RR-ONLY LOGISTIC REGRESSION ON REBALANCED FOLDS):")
print(f"  Pooled Macro F1 {{N, S, V}}: {macro_f1_nsv_oof:.4f}")
print(f"  Mean-of-Folds Macro F1:    {mean_of_folds_f1:.4f} +/- {std_of_folds_f1:.4f}")
print(f"  Accuracy:                  {rep_oof['accuracy']*100:.2f}%")
print(f"  Class N F1:                {rep_oof['N']['f1-score']:.4f}")
print(f"  Class S F1:                {rep_oof['S']['f1-score']:.4f} | Sensitivity (Recall): {rep_oof['S']['recall']*100:.2f}% | Precision: {rep_oof['S']['precision']*100:.2f}%")
print(f"  Class V F1:                {rep_oof['V']['f1-score']:.4f} | Sensitivity (Recall): {rep_oof['V']['recall']*100:.2f}% | Precision: {rep_oof['V']['precision']*100:.2f}%")
print(f"  Class F F1:                {rep_oof['F']['f1-score']:.4f} | Sensitivity (Recall): {rep_oof['F']['recall']*100:.2f}% | Precision: {rep_oof['F']['precision']*100:.2f}%")
print(f"  Pooled False Alarm Rate:   {far_oof*100:.2f}% ({fa_oof} / {n_tot_oof})")
print(f"  Mean-of-Folds FAR:         {mean_far*100:.2f}%")
print("\nPooled Confusion Matrix (Rows=True [N,S,V,F], Cols=Pred [N,S,V,F]):")
for r_idx, row in enumerate(cm_oof):
    print(f"  {CLASS_NAMES[r_idx]}: {row}")
