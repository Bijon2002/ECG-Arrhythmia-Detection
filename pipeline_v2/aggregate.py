import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import glob
import json
import numpy as np
from metrics import CLASSES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="weighted_rr", help="Tag of the main model")
a = ap.parse_args()

def load_runs(tag):
    pattern = os.path.join(RESULTS_DIR, f"{tag}_run[0-9]*.json")
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        with open(f, "r") as fp:
            runs.append(json.load(fp))
    return runs

def pm(vals, scale=1.0, nd=3):
    v = np.array(vals, dtype=float) * scale
    if len(v) > 1:
        return f"{v.mean():.{nd}f} $\\pm$ {v.std(ddof=1):.{nd}f}"
    elif len(v) == 1:
        return f"{v.mean():.{nd}f}"
    else:
        return "--"

def col(runs, f):
    return [f(r) for r in runs]

M = load_runs(a.main)
if not M:
    print(f"Warning: No results found for main tag '{a.main}' in {RESULTS_DIR}")

L = [f"% Main model tag: {a.main} (Runs found: {len(M)})"]

# 1. Dataset Split Table
split_counts_path = os.path.join(DATA_DIR, "split_counts.json")
if os.path.exists(split_counts_path):
    with open(split_counts_path, "r") as f:
        c = json.load(f)
    L.append("\n% ---- Table: Dataset Class Distribution (AAMI EC57) ----")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption{Heartbeat Distribution Across Inter-Patient Partitions (DS1 / DS2)}")
    L.append("\\label{tab:split}")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\toprule")
    L.append("Class Description & Train (DS1) & Val (DS1) & Test (DS2) & Total \\\\")
    L.append("\\midrule")
    class_long_names = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Paced / Unknown (Q)"]
    for i in range(len(CLASSES)):
        tr_cnt = c['train'][i]
        va_cnt = c['val'][i]
        te_cnt = c['test'][i]
        tot = tr_cnt + va_cnt + te_cnt
        L.append(f"{class_long_names[i]} & {tr_cnt:,} & {va_cnt:,} & {te_cnt:,} & {tot:,} \\\\")
    L.append("\\midrule")
    L.append(f"Total & {sum(c['train']):,} & {sum(c['val']):,} & {sum(c['test']):,} & {sum(c['train']) + sum(c['val']) + sum(c['test']):,} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

# 2. Per-Class Benchmark Table (Main Model)
if M:
    L.append("\n% ---- Table: Per-Class Diagnostic Performance (Test Set DS2) ----")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption{Per-Class Evaluation of the Proposed Calibrated Deep Ensemble on Unseen Patients (DS2)}")
    L.append("\\label{tab:perclass}")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\toprule")
    L.append("Arrhythmia Class & Precision & Recall (Sens.) & F1-Score & Support \\\\")
    L.append("\\midrule")
    sup = M[0]["ensemble"]["support"]
    for i, nme in enumerate(CLASSES):
        if sup[i] == 0:
            L.append(f"{nme} & -- & -- & -- & 0 \\\\")
            continue
        P = pm(col(M, lambda r: r["ensemble"]["precision"][i]))
        R = pm(col(M, lambda r: r["ensemble"]["recall"][i]))
        F = pm(col(M, lambda r: r["ensemble"]["f1"][i]))
        L.append(f"{nme} & {P} & {R} & {F} & {sup[i]:,} \\\\")
    L.append("\\midrule")
    macro_f1_val = pm(col(M, lambda r: r["ensemble"]["macro_f1"]))
    overall_acc = pm(col(M, lambda r: r["ensemble"]["acc"]), scale=100.0, nd=2)
    L.append(f"Macro Average & -- & -- & {macro_f1_val} & {sum(sup):,} \\\\")
    L.append(f"Overall Accuracy & \\multicolumn{{4}}{{c}}{{{overall_acc}\\%}} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

# 3. Ablation Study Table
def get_abl_row(name, tag, key):
    runs = load_runs(tag)
    if not runs:
        return f"{name} & -- & -- & -- & -- \\\\"
    acc_str = pm(col(runs, lambda r: r[key]["acc"]), scale=100.0, nd=1)
    mf1_str = pm(col(runs, lambda r: r[key]["macro_f1"]))
    f1_s_str = pm(col(runs, lambda r: r[key]["f1"][1]))
    f1_f_str = pm(col(runs, lambda r: r[key]["f1"][3]))
    return f"{name} & {acc_str} & {mf1_str} & {f1_s_str} & {f1_f_str} \\\\"

L.append("\n% ---- Table: Architectural Ablation Study ----")
L.append("\\begin{table}[t]")
L.append("\\centering")
L.append("\\caption{Ablation Study on Inter-Patient Test Set (DS2): Impact of Weighting, Ensembling, and RR Features}")
L.append("\\label{tab:ablation}")
L.append("\\begin{tabular}{lcccc}")
L.append("\\toprule")
L.append("Architecture & Accuracy (\\%) & Macro F1 & F1 ($S$) & F1 ($F$) \\\\")
L.append("\\midrule")
L.append(get_abl_row("Single 1D-CNN (Unweighted)", "unweighted", "single"))
L.append(get_abl_row("Single 1D-CNN (Weighted)", "weighted", "single"))
L.append(get_abl_row("Single 1D-CNN (Weighted + RR)", "weighted_rr", "single"))
L.append(get_abl_row("Deep Ensemble (Unweighted)", "unweighted", "ensemble"))
L.append(get_abl_row("Deep Ensemble (Weighted)", "weighted", "ensemble"))
L.append(get_abl_row("Deep Ensemble + RR (Proposed)", "weighted_rr", "ensemble"))
L.append("\\bottomrule")
L.append("\\end{tabular}")
L.append("\\end{table}")

# 4. Data Leakage Comparison Table
L.append("\n% ---- Table: Data Leakage Comparison (Beat-Wise vs Inter-Patient) ----")
L.append("\\begin{table}[t]")
L.append("\\centering")
L.append("\\caption{Data Leakage Demonstration: Identical Proposed Architecture Evaluated Under Two Split Regimes}")
L.append("\\label{tab:leakage}")
L.append("\\begin{tabular}{lcccc}")
L.append("\\toprule")
L.append("Partitioning Paradigm & Accuracy (\\%) & Macro F1 & F1 ($S$) & F1 ($F$) \\\\")
L.append("\\midrule")
bw_runs = load_runs(a.main + "_bw")
L.append(f"{get_abl_row('Beat-Wise Random Split (Contaminated)', a.main + '_bw', 'ensemble')}")
L.append(f"{get_abl_row('Inter-Patient DS1/DS2 Split (Realistic)', a.main, 'ensemble')}")
L.append("\\bottomrule")
L.append("\\end{tabular}")
L.append("\\end{table}")

# 5. Calibration Results Table
if M:
    L.append("\n% ---- Table: Probability Calibration (ECE / NLL / Brier) ----")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption{Post-hoc Temperature Scaling Calibration Impact on Test Set}")
    L.append("\\label{tab:calibration}")
    L.append("\\begin{tabular}{lcc}")
    L.append("\\toprule")
    L.append("Metric & Before Calibration (Raw) & After Temperature Scaling \\\\")
    L.append("\\midrule")
    for metric_name, k in [("Negative Log-Likelihood (NLL)", "nll"), ("Expected Calibration Error (ECE)", "ece"), ("Brier Score", "brier")]:
        raw_val = pm(col(M, lambda r: r["calibration"]["raw"][k]), scale=1.0, nd=4)
        cal_val = pm(col(M, lambda r: r["calibration"]["cal"][k]), scale=1.0, nd=4)
        L.append(f"{metric_name} & {raw_val} & {cal_val} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

# 6. Uncertainty Error Detection Table
if M:
    names = {
        "softmax": "Softmax Confidence",
        "MCvar": "MC Dropout Variance",
        "PE": "Predictive Entropy",
        "CBE": "Cluster-Based Entropy (Ours)"
    }
    L.append("\n% ---- Table: Uncertainty Error Detection Performance ----")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption{Uncertainty Metric Benchmark for Detecting Diagnostic Errors}")
    L.append("\\label{tab:uncertainty}")
    L.append("\\begin{tabular}{lcccc}")
    L.append("\\toprule")
    L.append("Uncertainty Formulation & AUROC & AUPRC & AURC & Retained Acc @ 10\\% Referral \\\\")
    L.append("\\midrule")
    for k, nme in names.items():
        auroc_s = pm(col(M, lambda r: r["uncertainty"][k]["auroc"]))
        auprc_s = pm(col(M, lambda r: r["uncertainty"][k]["auprc"]))
        aurc_s = pm(col(M, lambda r: r["uncertainty"][k]["aurc"]), scale=1.0, nd=4)
        acc10_s = pm(col(M, lambda r: r["uncertainty"][k]["acc@10"]), scale=100.0, nd=1)
        L.append(f"{nme} & {auroc_s} & {auprc_s} & {aurc_s} & {acc10_s}\\% \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

# 7. Referral Rate Sweeps Table
if M:
    L.append("\n% ---- Table: Referral Sweeps (Accuracy / Macro F1 on Kept Beats) ----")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption{Selective Classification: Diagnostic Performance on Retained Beats Across Referral Rates}")
    L.append("\\label{tab:sweeps}")
    L.append("\\begin{tabular}{lccccc}")
    L.append("\\toprule")
    L.append("Uncertainty Metric & 5\\% Referral & 10\\% Referral & 15\\% Referral & 20\\% Referral & 30\\% Referral \\\\")
    L.append("\\midrule")
    for k, nme in names.items():
        cells = []
        for rr in (5, 10, 15, 20, 30):
            acc_str = pm(col(M, lambda r: r["uncertainty"][k][f"acc@{rr}"]), scale=100.0, nd=1)
            mf_str = pm(col(M, lambda r: r["uncertainty"][k][f"mf1@{rr}"]), scale=1.0, nd=2)
            cells.append(f"{acc_str}\\% / {mf_str}")
        L.append(f"{nme} & " + " & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

output_tex_path = os.path.join(RESULTS_DIR, "tables.tex")
with open(output_tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print(f"\nSuccessfully generated publication LaTeX tables to: {output_tex_path}")
print("Preview of generated tables:\n")
print("\n".join(L[:60]))
