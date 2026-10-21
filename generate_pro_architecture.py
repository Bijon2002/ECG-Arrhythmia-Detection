import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

# Create high-res publication canvas (300 DPI)
fig = plt.figure(figsize=(18, 9.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1800)
ax.set_ylim(0, 960)
ax.axis('off')

# Clean crisp background
bg = patches.Rectangle((0, 0), 1800, 960, facecolor='#FFFFFF')
ax.add_patch(bg)

# Top Banner
title_box = FancyBboxPatch((30, 895), 1740, 50, boxstyle="round,pad=0,rounding_size=8",
                           facecolor="#1A365D", edgecolor="#0D233A", linewidth=1.5)
ax.add_patch(title_box)
ax.text(900, 920, "PROPOSED UNCERTAINTY-AWARE DEEP ENSEMBLE ARCHITECTURE & SELECTIVE TRIAGE PIPELINE",
        ha='center', va='center', fontsize=14.5, fontweight='bold', color='#FFFFFF', family='sans-serif')

def draw_container(x, y, w, h, bg_color, border_color, title_text, tag_color):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                          facecolor=bg_color, edgecolor=border_color, linewidth=2.0)
    ax.add_patch(card)
    
    banner_h = 36
    banner = FancyBboxPatch((x, y + h - banner_h), w, banner_h, boxstyle="round,pad=0,rounding_size=12",
                            facecolor=tag_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(banner)
    sq_patch = patches.Rectangle((x, y + h - banner_h), w, 12, facecolor=tag_color, edgecolor='none')
    ax.add_patch(sq_patch)
    
    ax.text(x + w / 2, y + h - 18, title_text, ha='center', va='center',
            fontsize=11.5, fontweight='bold', color='#FFFFFF', family='sans-serif')

def draw_subcard(x, y, w, h, title, items, bg_color="#FFFFFF", border_color="#B0BEC5", title_color="#263238"):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                          facecolor=bg_color, edgecolor=border_color, linewidth=1.2)
    ax.add_patch(card)
    
    curr_y = y + h - 15
    if title:
        ax.text(x + 10, curr_y, title, ha='left', va='center',
                fontsize=9.8, fontweight='bold', color=title_color, family='sans-serif')
        curr_y -= 17
    
    for item in items:
        ax.text(x + 10, curr_y, item, ha='left', va='center',
                fontsize=8.3, color="#37474F", family='sans-serif')
        curr_y -= 15.5

def draw_arrow(x1, y1, x2, y2, color="#455A64", lw=2.2, label=None, label_pos=None, connectionstyle=None):
    if connectionstyle:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                                color=color, linewidth=lw, connectionstyle=connectionstyle)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                                color=color, linewidth=lw)
    ax.add_patch(arrow)
    if label and label_pos:
        ax.text(label_pos[0], label_pos[1], label, ha='center', va='center',
                fontsize=8.2, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec=color, lw=0.8, alpha=0.95))

# ==============================================================================
# 1. MODULE 1: DATASET & PARTITIONING (GREEN)
# ==============================================================================
draw_container(30, 440, 310, 435, "#E8F5E9", "#2E7D32", "1. DATA & COHORT SPLIT", "#2E7D32")

draw_subcard(45, 745, 280, 85, "MIT-BIH Database", [
    "• 48 Dual-Lead ECG Records (360 Hz)",
    "• AAMI Superclasses: {N, S, V, F, Q}",
    "• 100,682 Annotated Heartbeats"
], border_color="#A5D6A7", title_color="#1B5E20")

draw_subcard(45, 625, 280, 105, "Inter-Patient Protocol", [
    "• de Chazal DS1/DS2 Record Partition",
    "• Eliminates Beat-Wise Data Leakage",
    "• 44 Records (Excl. Paced 102,104,107,217)",
    "• Note: Recs 201/202 Share One Subject"
], border_color="#A5D6A7", title_color="#1B5E20")

draw_subcard(45, 535, 280, 75, "DS1: Training Cohort (22 Recs)", [
    "• Train: 42,239 beats (18 records)",
    "• Val: 8,756 beats (4 records)",
    "• Used for Model Weights & Temperature"
], bg_color="#E8F8F5", border_color="#A3E4D7", title_color="#117864")

draw_subcard(45, 455, 280, 65, "DS2: Test Cohort (22 Recs)", [
    "• Test: 49,687 beats (Unseen patients)",
    "• Unbiased Generalization Evaluation"
], bg_color="#E8F8F5", border_color="#A3E4D7", title_color="#117864")

# ==============================================================================
# 2. MODULE 2: DUAL-BRANCH PREPROCESSING (AMBER)
# ==============================================================================
draw_container(370, 440, 320, 435, "#FFF3E0", "#E65100", "2. DUAL PREPROCESSING", "#E65100")

draw_subcard(385, 665, 290, 165, "Branch A: Waveform Preprocessing", [
    "• Lead MLII Electrocardiogram Channel",
    "• 2nd-Order Butterworth Bandpass (0.5–45 Hz)",
    "• Centered R-Peak Window: [-180, +180]",
    "• Fixed 360 Samples (1.0 s Beat Duration)",
    "• Z-Score Normalization: (x - μ) / (σ + ε)",
    "• Morphology Tensor: x_i ∈ R^{360}"
], border_color="#FFCC80", title_color="#E65100")

draw_subcard(385, 455, 290, 195, "Branch B: Auxiliary RR Timing", [
    "• Pan-Tompkins Peak Detection Cues",
    "• Resolves S vs N Morphological Ambiguity:",
    "  - Pre-RR: Δt_pre = (s_i - s_{i-1}) / 360",
    "  - Post-RR: Δt_post = (s_{i+1} - s_i) / 360",
    "  - Local RR: Mean of prior 10 intervals",
    "  - Ratio: r_pre = Δt_pre / Local_RR",
    "• Timing Feature Vector: r_i ∈ R^4",
    "• Injected directly at Dense layer"
], border_color="#FFCC80", title_color="#E65100")

# ==============================================================================
# 3. MODULE 3: 1D-CNN BACKBONE & FUSION (BLUE)
# ==============================================================================
draw_container(720, 440, 330, 435, "#E3F2FD", "#1565C0", "3. 1D-CNN & FEATURE FUSION", "#1565C0")

draw_subcard(735, 695, 300, 135, "1D-CNN Feature Extractor", [
    "• Block 1: Conv1D(32, k=5) + BN + ReLU + MaxP",
    "• Block 2: Conv1D(64, k=5) + BN + ReLU + MaxP",
    "• Block 3: Conv1D(128, k=5) + BN + ReLU + MaxP",
    "• Flattened Feature Map: 128 × 45",
    "• Waveform Representation: 5,760 dims"
], border_color="#90CAF9", title_color="#0D47A1")

draw_subcard(735, 595, 300, 85, "Dual-Branch Concatenation", [
    "• Morphological Features: 5,760-d",
    "• Auxiliary RR Timing Features: 4-d",
    "• Joint Representation: h ∈ R^{5,764}"
], bg_color="#E1F5FE", border_color="#81D4FA", title_color="#01579B")

draw_subcard(735, 455, 300, 125, "Classifier & Loss Optimization", [
    "• Dense Layer 1: Linear(5,764 -> 256) + ReLU",
    "• Dropout Layer: Rate p = 0.5 Regularization",
    "• Dense Layer 2: Linear(256 -> 5 Classes)",
    "• Class-Weighted Loss: w_c = N / (5 × N_c)",
    "• Adam Optimizer (lr=1e-3, batch=128)"
], border_color="#90CAF9", title_color="#0D47A1")

# ==============================================================================
# 4. MODULE 4: CALIBRATED DEEP ENSEMBLE (RED)
# ==============================================================================
draw_container(1080, 440, 330, 435, "#FFEBEE", "#C62828", "4. CALIBRATED DEEP ENSEMBLE", "#C62828")

draw_subcard(1095, 715, 300, 115, "5-Member Deep Ensemble", [
    "• 5 Independently Seeded 1D-CNNs (Seeds 0–4)",
    "• Diversity via Random Weight Initializations",
    "• Smooths Non-Convex Decision Landscape",
    "• Produces 5 Raw Logit Vectors: {z_1, ..., z_5}"
], border_color="#EF9A9A", title_color="#B71C1C")

draw_subcard(1095, 590, 300, 110, "Post-Hoc Temperature Scaling", [
    "• Per-Member Learned Temperature T_m > 0",
    "• NLL Minimization on DS1 Validation Set",
    "• Learned: T = [0.836, 1.183, 1.487, 0.979, 1.325]",
    "• Calibration: ECE drops 4.28% -> 3.04% (-29%)",
    "• Scaled Logits: z~_m = z_m / T_m"
], bg_color="#FFEBEE", border_color="#EF9A9A", title_color="#B71C1C")

draw_subcard(1095, 455, 300, 120, "Logit Averaging & Posterior", [
    "• Ensemble Logit Averaging (Eq. 5):",
    "    z_ens = (1/M) Σ_{m=1}^M (z_m / T_m)",
    "• Calibrated Posterior: p_ens = Softmax(z_ens)",
    "• Unseen Test Accuracy: 85.86% (DS2)",
    "• Macro F1: 0.371 (5-class) / 0.463 (4-class)"
], bg_color="#FFCDD2", border_color="#E57373", title_color="#B71C1C")

# ==============================================================================
# 5. MODULE 5: UNCERTAINTY QUANTIFICATION (PURPLE)
# ==============================================================================
draw_container(1440, 440, 330, 435, "#F3E5F5", "#6A1B9A", "5. UQ & RISK QUANTIFICATION", "#6A1B9A")

draw_subcard(1455, 735, 300, 95, "1. Softmax Confidence (SC)", [
    "• Equation: u_SC = 1 - max_c p_ens(c | x)",
    "• Superior Empirical Error Detector",
    "• AUROC: 0.819 | AUPRC: 0.405 | AURC: 0.0465",
    "• Zero Extra Inference Computational Cost"
], border_color="#CE93D8", title_color="#4A148C")

draw_subcard(1455, 635, 300, 85, "2. Predictive Entropy (PE)", [
    "• Equation: u_PE = -Σ p_ens log2(p_ens)",
    "• Information-Theoretic Dispersal Score",
    "• AUROC: 0.810 | AUPRC: 0.361"
], border_color="#CE93D8", title_color="#4A148C")

draw_subcard(1455, 525, 300, 95, "3. Cluster-Based Entropy (CBE)", [
    "• Merged into 4 Clinical Severity Tiers:",
    "    p_clust = [p_N,  p_S,  p_{V+F},  p_Q]^T",
    "    u_CBE = -Σ p_clust log2(p_clust)",
    "• AUROC: 0.798 | Clinically Grouped Tiers"
], bg_color="#EDE7F6", border_color="#B39DDB", title_color="#311B92")

draw_subcard(1455, 455, 300, 55, "4. MC-Dropout Variance (MCD)", [
    "• Epistemic Variance over 20 Stochastic Passes",
    "• AUROC: 0.754 | Higher Computational Latency"
], border_color="#CE93D8", title_color="#4A148C")

# Top Connecting Arrows (Smooth Left-to-Right)
draw_arrow(340, 657, 370, 657, color="#2E7D32", lw=2.4)
draw_arrow(690, 747, 720, 747, color="#E65100", lw=2.4)
draw_arrow(690, 552, 720, 637, color="#E65100", lw=2.4)
draw_arrow(1050, 657, 1080, 657, color="#1565C0", lw=2.4)
draw_arrow(1410, 657, 1440, 657, color="#C62828", lw=2.4)

# ==============================================================================
# 6. MODULE 6: LOWER WORKFLOW: CLINICAL SELECTIVE CLASSIFICATION & TRIAGE
# ==============================================================================
draw_container(30, 25, 1740, 385, "#F5F5F5", "#37474F", "6. RISK-STRATIFIED SELECTIVE CLASSIFICATION & HUMAN-IN-THE-LOOP CLINICAL TRIAGE", "#37474F")

# Card 1 (Far Left, x=50, w=410): Key Clinical & Scientific Contributions
draw_subcard(50, 45, 410, 315, "Key Clinical & Scientific Contributions", [
    "1. Inter-Patient Integrity: Rigorously benchmarks on 49,687",
    "   beats from unseen records, exposing real-world difficulty.",
    "2. Auxiliary Timing Fusion: 4 RR timing metrics compensate",
    "   for S-beat morphological mimicry of normal beats.",
    "3. Trustworthy Calibration: Per-member temperature scaling",
    "   reduces ECE by 29% (4.28% -> 3.04%) for dependable probabilities.",
    "4. Proven Error Detection: Softmax Confidence achieves 0.819",
    "   AUROC, effectively flagging misclassified beats.",
    "5. Collaborative Triage: 10% referral raises accuracy to 89.6%,",
    "   and 30% referral achieves 94.7% for safe clinical deployment."
], bg_color="#FFFFFF", border_color="#5C6BC0", title_color="#1A237E")

# Card 2 (Left-Center, x=480, w=430): Selective Classification Policy & Empirical Sweep
draw_subcard(480, 45, 430, 315, "Selective Classification Policy (Threshold τ)", [
    "• Referral Budget r ∈ [0.05, 0.30] sets Uncertainty Threshold τ",
    "• Mathematical Decision Rule:",
    "     y*(x) = Argmax_c p_ens(c|x)    if u(x) ≤ τ  (Autonomous)",
    "     y*(x) = REFER_TO_CLINICIAN      if u(x) > τ  (Referred)",
    "• Empirical Sweeps on 49,687 Unseen Test Beats (DS2):",
    "   - 0% Referral (Raw Model):       85.86% Acc | Macro F1: 0.371",
    "   - 5% Referral to Doctor:         87.80% Acc | Macro F1: 0.380",
    "   - 10% Referral (Optimal Tradeoff): 89.60% Acc | Macro F1: 0.390",
    "   - 20% Referral to Doctor:        92.40% Acc | Macro F1: 0.400",
    "   - 30% Referral (High Precision): 94.70% Acc | Macro F1: 0.420",
    "• Reduces cardiologist manual review workload by 70% to 90%"
], bg_color="#FFFFFF", border_color="#78909C", title_color="#263238")

# Card 3 (Center-Right): The Two Outcome Pathways
# Upper Card (Green: Autonomous)
auto_card = FancyBboxPatch((930, 210), 450, 150, boxstyle="round,pad=0,rounding_size=10",
                           facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.8)
ax.add_patch(auto_card)
ax.text(950, 335, "✓ Autonomous Operational Diagnosis", fontsize=11, fontweight='bold', color="#1B5E20")
ax.text(950, 308, "• High-confidence automated beat classification (u(x) ≤ τ)", fontsize=8.6, color="#2E7D32")
ax.text(950, 286, "• Automated electronic medical record (EMR) telemetry logging", fontsize=8.6, color="#2E7D32")
ax.text(950, 264, "• Primary Classes: Normal (N) & Ventricular Ectopic (V)", fontsize=8.6, color="#1B5E20")
ax.text(950, 238, "• Retained Beat Accuracy: 89.6% (r=0.10) to 94.7% (r=0.30)", fontsize=8.8, fontweight='bold', color="#1B5E20")

# Lower Card (Red: Expert Referral)
doc_card = FancyBboxPatch((930, 45), 450, 150, boxstyle="round,pad=0,rounding_size=10",
                          facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.8)
ax.add_patch(doc_card)
ax.text(950, 170, "⚠ Cardiologist Expert Referral (Human-in-the-Loop)", fontsize=11, fontweight='bold', color="#B71C1C")
ax.text(950, 143, "• Routes ambiguous & high-risk morphologies (u(x) > τ)", fontsize=8.6, color="#C62828")
ax.text(950, 121, "• Intercepts difficult Supraventricular (S) & Fusion (F) beats", fontsize=8.6, color="#C62828")
ax.text(950, 99, "• Eliminates diagnostic oversight and fatal misclassifications", fontsize=8.6, color="#B71C1C")
ax.text(950, 73, "• Workload Optimization: Automated 90%, Doctor Reviews 10%", fontsize=8.8, fontweight='bold', color="#B71C1C")

# Card 4 (Far Right): Uncertainty Decision Diamond directly under Module 5
diamond_x, diamond_y = 1585, 202
diamond = Polygon([[diamond_x, diamond_y + 55],
                   [diamond_x + 95, diamond_y],
                   [diamond_x, diamond_y - 55],
                   [diamond_x - 95, diamond_y]],
                  closed=True, facecolor="#FFF9C4", edgecolor="#FBC02D", linewidth=2.0)
ax.add_patch(diamond)
ax.text(diamond_x, diamond_y + 12, "Uncertainty Gate", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#F57F17")
ax.text(diamond_x, diamond_y - 10, "u(x) ≤ τ ?", ha='center', va='center', fontsize=11, fontweight='bold', color="#263238")

# Arrow dropping from Module 5 into Decision Diamond
draw_arrow(1585, 440, 1585, diamond_y + 55, color="#6A1B9A", lw=2.5, label="Beat Posterior & Score u(x)", label_pos=(1585, 340))

# Path 1: YES branch (Green) from diamond left to Autonomous Diagnosis Card
draw_arrow(diamond_x - 95, diamond_y + 20, 1380, 285, color="#2E7D32", lw=2.4,
           label="YES (Confident: u(x) ≤ τ)", label_pos=(1475, 290), connectionstyle="arc3,rad=-0.1")

# Path 2: NO branch (Red) from diamond left to Clinical Referral Card
draw_arrow(diamond_x - 95, diamond_y - 20, 1380, 120, color="#C62828", lw=2.4,
           label="NO (Uncertain: u(x) > τ)", label_pos=(1475, 115), connectionstyle="arc3,rad=0.1")

# Output paths
output_paths = [
    "f:/ECG-Arrhythmia-Detection/fig1_architecture.png",
    "f:/ECG-Arrhythmia-Detection/overleaf_figures/fig1_architecture.png",
    "f:/ECG-Arrhythmia-Detection/results/plots/ensemble_architecture_fig3_5.png"
]

for p in output_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f"Saved: {p}")

plt.close()
print("Professional architecture diagram generated successfully!")
