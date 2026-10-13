import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

CLASSES = ['N', 'S', 'V', 'F', 'Q']

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def cbe(p, eps=1e-12):
    """
    Cluster-Based Entropy (4 Clinical Risk Tiers):
    Tier 0: Normal {N} -> p[:, 0]
    Tier 1: Supraventricular {S} -> p[:, 1]
    Tier 2: Ventricular & Fusion {V, F} -> p[:, 2] + p[:, 3]
    Tier 3: Unknown / Paced {Q} -> p[:, 4]
    """
    p_clust = np.stack([
        p[:, 0],
        p[:, 1],
        p[:, 2] + p[:, 3],
        p[:, 4]
    ], axis=1)
    p_clust = np.clip(p_clust, eps, 1.0)
    return -np.sum(p_clust * np.log2(p_clust), axis=1)

def clf_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    support = cm.sum(axis=1)
    acc = float(np.sum(np.diag(cm)) / max(np.sum(cm), 1))

    precision, recall, f1 = [], [], []
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(score))

    # Compute macro F1 over classes that have support > 0
    present = [i for i in range(len(CLASSES)) if support[i] > 0]
    macro_f1 = float(np.mean([f1[i] for i in present])) if present else 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": [int(s) for s in support],
        "cm": cm.tolist()
    }

def nll(p, y, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))

def brier(p, y):
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

def ece(p, y, n_bins=15):
    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    accuracies = (predictions == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    N = len(y)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (confidences > lo) & (confidences <= hi) if b < n_bins - 1 \
               else (confidences >= lo) & (confidences <= hi)
        bin_count = mask.sum()
        if bin_count > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece_val += (bin_count / N) * abs(bin_acc - bin_conf)

    return float(ece_val)

def error_detection(scores, errors, refs=(0.05, 0.10, 0.15, 0.20, 0.30)):
    """
    Evaluates uncertainty scores for error detection.
    scores: higher score = higher uncertainty (should predict error=1)
    errors: 1 if prediction was wrong, 0 if correct
    """
    has_both_classes = len(np.unique(errors)) > 1
    auroc = float(roc_auc_score(errors, scores)) if has_both_classes else 0.5
    auprc = float(average_precision_score(errors, scores)) if has_both_classes else float(errors.mean())

    order = np.argsort(scores, kind="stable")
    N = len(errors)
    coverage = np.arange(1, N + 1) / N
    # Cumulative error rate on kept (lowest uncertainty) beats
    sorted_err = errors[order]
    cum_err = np.cumsum(sorted_err) / np.arange(1, N + 1)
    aurc = float(np.trapz(cum_err, coverage))

    results = {
        "auroc": auroc,
        "auprc": auprc,
        "aurc": aurc
    }

    # Accuracy on kept beats at referral cutoffs
    for r in refs:
        keep_n = int(round((1 - r) * N))
        acc_kept = float(1.0 - sorted_err[:keep_n].mean()) if keep_n > 0 else 1.0
        results[f"acc@{int(round(r * 100))}"] = acc_kept

    return results, {"coverage": coverage.tolist(), "risk": cum_err.tolist()}
