import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
import numpy as np

# Create figure canvas matching the reference diagram's aspect ratio and layout
fig = plt.figure(figsize=(19, 13.5), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1900)
ax.set_ylim(0, 1350)
ax.axis('off')

# Clean white background
bg = patches.Rectangle((0, 0), 1900, 1350, facecolor='#FFFFFF')
ax.add_patch(bg)

# -------------------------------------------------------------------------
# Helper Functions: Containers, Cards, and Arrows
# -------------------------------------------------------------------------
def draw_rounded_container(x, y, w, h, bg_color, border_color, title_text="", lw=2.0, title_color="#212121", title_size=13):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16",
                          facecolor=bg_color, edgecolor=border_color, linewidth=lw)
    ax.add_patch(card)
    if title_text:
        # Centered bold title at the top, just like reference image
        ax.text(x + w / 2, y + h - 22, title_text, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=title_color, family='sans-serif')

def draw_arrow(x1, y1, x2, y2, color="#7F8C8D", lw=2.2, label=None, label_pos=None, connectionstyle=None):
    if connectionstyle:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15,
                                color=color, linewidth=lw, connectionstyle=connectionstyle)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15,
                                color=color, linewidth=lw)
    ax.add_patch(arrow)
    if label and label_pos:
        ax.text(label_pos[0], label_pos[1], label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="none", alpha=0.9))

# -------------------------------------------------------------------------
# Vector Icon Drawing Helpers (Crisp geometric iconography)
# -------------------------------------------------------------------------
def draw_icon_ecg(cx, cy, size=18, color="#2E7D32"):
    pts = np.array([
        [cx - size, cy],
        [cx - size*0.5, cy],
        [cx - size*0.35, cy + size*0.4],
        [cx - size*0.2, cy - size*0.3],
        [cx, cy + size*0.9],
        [cx + size*0.2, cy - size*0.8],
        [cx + size*0.4, cy + size*0.3],
        [cx + size*0.6, cy],
        [cx + size, cy]
    ])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.2, solid_capstyle='round')

def draw_icon_camera_patient(cx, cy, size=15, color="#2E7D32"):
    rect = FancyBboxPatch((cx - size, cy - size*0.7), size*2, size*1.4, boxstyle="round,pad=0,rounding_size=3",
                          facecolor="none", edgecolor=color, lw=1.8)
    ax.add_patch(rect)
    ax.add_patch(Circle((cx, cy), size*0.45, facecolor="none", edgecolor=color, lw=1.8))
    ax.plot([cx + size*0.5, cx + size*0.7], [cy + size*0.5, cy + size*0.5], color=color, lw=2.0)

def draw_icon_doctor(cx, cy, size=16, color="#2E7D32"):
    ax.add_patch(Circle((cx, cy + size*0.45), size*0.4, facecolor="none", edgecolor=color, lw=1.8))
    body = patches.Arc((cx, cy - size*0.5), size*1.8, size*1.4, theta1=20, theta2=160, edgecolor=color, lw=1.8)
    ax.add_patch(body)
    # Stethoscope / plus
    ax.plot([cx - size*0.2, cx + size*0.2], [cy - size*0.1, cy - size*0.1], color=color, lw=1.6)
    ax.plot([cx, cx], [cy - size*0.3, cy + size*0.1], color=color, lw=1.6)

def draw_icon_rgb(cx, cy, size=14):
    ax.add_patch(Circle((cx - size*0.4, cy + size*0.2), size*0.45, facecolor="#E74C3C", edgecolor="none", alpha=0.85))
    ax.add_patch(Circle((cx + size*0.4, cy + size*0.2), size*0.45, facecolor="#2ECC71", edgecolor="none", alpha=0.85))
    ax.add_patch(Circle((cx, cy - size*0.35), size*0.45, facecolor="#3498DB", edgecolor="none", alpha=0.85))

def draw_icon_filter(cx, cy, size=16, color="#D4AC0D"):
    rect = FancyBboxPatch((cx - size, cy - size*0.6), size*2, size*1.2, boxstyle="round,pad=0,rounding_size=4",
                          facecolor="none", edgecolor=color, lw=1.8)
    ax.add_patch(rect)
    x = np.linspace(cx - size*0.8, cx + size*0.8, 30)
    y = cy + size*0.4 * np.sin((x - cx)*0.3)
    ax.plot(x, y, color=color, lw=1.8)

def draw_icon_split(cx, cy, size=15, color="#D4AC0D"):
    # Pie chart with slice
    wedge1 = patches.Wedge((cx, cy), size, 45, 360, facecolor="#F39C12", edgecolor="none")
    wedge2 = patches.Wedge((cx + 3, cy + 3), size, 0, 45, facecolor="#E74C3C", edgecolor="none")
    ax.add_patch(wedge1)
    ax.add_patch(wedge2)

def draw_icon_norm(cx, cy, size=15, color="#D4AC0D"):
    x = np.linspace(cx - size, cx + size, 30)
    y = cy - size*0.5 + size * np.exp(-((x - cx)/(size*0.4))**2)
    ax.plot(x, y, color=color, lw=2.0)
    ax.plot([cx - size, cx + size], [cy - size*0.5, cy - size*0.5], color=color, lw=1.5)

def draw_icon_network(cx, cy, size=16, color="#2980B9"):
    # Small neural net glyph
    nodes_l1 = [(cx - size*0.7, cy + size*0.5), (cx - size*0.7, cy), (cx - size*0.7, cy - size*0.5)]
    nodes_l2 = [(cx, cy + size*0.3), (cx, cy - size*0.3)]
    nodes_l3 = [(cx + size*0.7, cy)]
    for n1 in nodes_l1:
        for n2 in nodes_l2:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color="#BDC3C7", lw=1.0)
    for n2 in nodes_l2:
        for n3 in nodes_l3:
            ax.plot([n2[0], n3[0]], [n2[1], n3[1]], color="#BDC3C7", lw=1.0)
    for n in nodes_l1 + nodes_l2 + nodes_l3:
        ax.add_patch(Circle(n, 3.2, facecolor=color, edgecolor="#FFFFFF", lw=0.8))

def draw_icon_transfer(cx, cy, size=15, color="#2980B9"):
    # Two overlapping shapes / arrows
    ax.add_patch(Circle((cx - size*0.35, cy), size*0.5, facecolor="none", edgecolor=color, lw=1.8))
    ax.add_patch(Circle((cx + size*0.35, cy), size*0.5, facecolor="none", edgecolor=color, lw=1.8))
    ax.plot([cx - size*0.1, cx + size*0.1], [cy, cy], color=color, lw=2.2)

def draw_icon_loss(cx, cy, size=15, color="#2980B9"):
    # Tuning fork / math symbol
    ax.plot([cx - size*0.6, cx], [cy + size*0.6, cy - size*0.4], color=color, lw=2.0)
    ax.plot([cx + size*0.6, cx], [cy + size*0.6, cy - size*0.4], color=color, lw=2.0)
    ax.plot([cx, cx], [cy - size*0.4, cy - size*0.8], color=color, lw=2.0)

def draw_icon_gear(cx, cy, size=15, color="#2980B9"):
    ax.add_patch(Circle((cx, cy), size*0.5, facecolor="none", edgecolor=color, lw=2.2))
    ax.add_patch(Circle((cx, cy), size*0.2, facecolor=color, edgecolor="none"))
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        gx = cx + size*0.65 * np.cos(angle)
        gy = cy + size*0.65 * np.sin(angle)
        ax.add_patch(Circle((gx, gy), 2.2, facecolor=color, edgecolor="none"))

def draw_icon_hand(cx, cy, size=15, color="#2980B9"):
    # Stop hand / palm
    ax.add_patch(Circle((cx, cy), size*0.6, facecolor=color, edgecolor="none", alpha=0.2))
    ax.add_patch(Circle((cx, cy), size*0.6, facecolor="none", edgecolor=color, lw=1.8))
    ax.plot([cx - size*0.3, cx + size*0.3], [cy, cy], color=color, lw=2.5)

def draw_icon_database(cx, cy, size=16, color="#C0392B"):
    # Cylinder database
    w = size * 1.8
    h = size * 1.6
    ax.add_patch(patches.Ellipse((cx, cy + h/2), w, h*0.35, facecolor="#FDEDEC", edgecolor=color, lw=1.8))
    ax.add_patch(Rectangle((cx - w/2, cy - h/2), w, h, facecolor="#FDEDEC", edgecolor="none"))
    ax.plot([cx - w/2, cx - w/2], [cy - h/2, cy + h/2], color=color, lw=1.8)
    ax.plot([cx + w/2, cx + w/2], [cy - h/2, cy + h/2], color=color, lw=1.8)
    ax.add_patch(patches.Arc((cx, cy), w, h*0.35, theta1=180, theta2=360, edgecolor=color, lw=1.8))
    ax.add_patch(patches.Ellipse((cx, cy - h/2), w, h*0.35, facecolor="#FDEDEC", edgecolor=color, lw=1.8))

def draw_icon_phone(cx, cy, size=16, color="#27AE60"):
    # Wearable Holter / Device
    w = size * 1.1
    h = size * 2.0
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0,rounding_size=4",
                          facecolor="none", edgecolor=color, lw=2.0)
    ax.add_patch(rect)
    ax.add_patch(Rectangle((cx - w/2 + 2, cy - h/2 + 5), w - 4, h - 10, facecolor="#E8F8F5", edgecolor="none"))
    # Pulse on screen
    ax.plot([cx - 4, cx - 2, cx, cx + 2, cx + 4], [cy, cy + 3, cy - 3, cy + 4, cy], color=color, lw=1.5)
    ax.add_patch(Circle((cx, cy - h/2 + 3), 1.5, facecolor=color, edgecolor="none"))

def draw_icon_trend(cx, cy, size=16, color="#F39C12"):
    ax.plot([cx - size*0.8, cx - size*0.3, cx + size*0.2, cx + size*0.8],
            [cy - size*0.6, cy - size*0.1, cy - size*0.4, cy + size*0.7], color=color, lw=2.2)
    ax.add_patch(Polygon([[cx + size*0.8, cy + size*0.7],
                          [cx + size*0.4, cy + size*0.7],
                          [cx + size*0.8, cy + size*0.3]], facecolor=color, edgecolor="none"))

def draw_icon_triage(cx, cy, size=16, color="#E67E22"):
    # Person with hands open / alert
    ax.add_patch(Circle((cx, cy + size*0.4), size*0.35, facecolor="none", edgecolor=color, lw=1.8))
    ax.plot([cx - size*0.6, cx, cx + size*0.6], [cy - size*0.1, cy - size*0.4, cy - size*0.1], color=color, lw=2.0)
    ax.add_patch(Circle((cx, cy + size*0.4), 2.5, facecolor=color, edgecolor="none"))

# =========================================================================
# TOP ROW CONTAINERS
# =========================================================================

# -------------------------------------------------------------------------
# Box 1 (Top-Left, Green): Multi-Source ECG Arrhythmia Datasets
# -------------------------------------------------------------------------
b1_x, b1_y, b1_w, b1_h = 50, 770, 420, 530
draw_rounded_container(b1_x, b1_y, b1_w, b1_h, "#EBF5EB", "#4CAF50", "Multi-Source ECG Arrhythmia Datasets", lw=2.2, title_color="#1B5E20")

# Sub-block 1: MIT-BIH Arrhythmia Database
draw_icon_ecg(90, 1220, size=18, color="#2E7D32")
ax.text(125, 1228, "MIT-BIH Arrhythmia Database", fontsize=11, fontweight='bold', color="#1B5E20")
ax.text(125, 1210, "• 48 Dual-Lead ECG Records (360 Hz)", fontsize=9.2, color="#2E7D32")
ax.text(125, 1192, "• 100,682 Validated Heartbeat Annotations", fontsize=9.2, color="#2E7D32")
ax.text(125, 1174, "• AAMI Superclasses: {N, S, V, F, Q}", fontsize=9.2, color="#2E7D32")

# Sub-block 2: Inter-Patient Evaluation Protocol
draw_icon_camera_patient(90, 1105, size=16, color="#2E7D32")
ax.text(125, 1118, "Inter-Patient Protocol (de Chazal)", fontsize=11, fontweight='bold', color="#1B5E20")
ax.text(125, 1100, "• Strict Record-Disjoint Split (DS1 / DS2)", fontsize=9.2, color="#2E7D32")
ax.text(125, 1082, "• Eliminates Intra-Patient Data Leakage", fontsize=9.2, color="#2E7D32")
ax.text(125, 1064, "• 44 Non-Paced Clinical Records Evaluated", fontsize=9.2, color="#2E7D32")
ax.text(125, 1046, "• Transparently Reports Recs 201/202 Overlap", fontsize=9.2, color="#2E7D32")

# Sub-block 3: Expert-Annotated Evaluation Cohorts
draw_icon_doctor(90, 970, size=18, color="#2E7D32")
ax.text(125, 988, "Expert-Annotated Cohorts", fontsize=11, fontweight='bold', color="#1B5E20")
ax.text(125, 970, "• DS1 Train: 42,239 beats (18 records)", fontsize=9.2, color="#2E7D32")
ax.text(125, 952, "• DS1 Val: 8,756 beats (4 records for T_m)", fontsize=9.2, color="#2E7D32")
ax.text(125, 934, "• DS2 Test: 49,687 beats (Unseen patients)", fontsize=9.2, color="#2E7D32")
ax.text(125, 916, "• Rigorous Benchmark for Real-World Triage", fontsize=9.2, color="#2E7D32")

# -------------------------------------------------------------------------
# Box 2 (Top-Center, Yellow): Data Preparation & Preprocessing
# -------------------------------------------------------------------------
b2_x, b2_y, b2_w, b2_h = 510, 770, 500, 530
draw_rounded_container(b2_x, b2_y, b2_w, b2_h, "#FEF9E7", "#F1C40F", "Data Preparation & Dual Preprocessing", lw=2.2, title_color="#7D6608")

# Grid Layout with Icons
# Row 1
draw_icon_rgb(550, 1220, size=16)
ax.text(580, 1226, "Lead Selection", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(580, 1208, "Lead MLII Channel", fontsize=9, color="#7D6608")

draw_icon_filter(770, 1220, size=16, color="#B7950B")
ax.text(805, 1226, "Butterworth Bandpass", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(805, 1208, "0.5 Hz – 45 Hz (Zero-Phase)", fontsize=9, color="#7D6608")

# Row 2
draw_icon_norm(550, 1140, size=16, color="#B7950B")
ax.text(580, 1146, "Z-Score Normalization", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(580, 1128, "(x - μ) / (σ + ε)", fontsize=9, color="#7D6608")

draw_icon_filter(770, 1140, size=16, color="#B7950B")
ax.text(805, 1146, "R-Peak Beat Slicing", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(805, 1128, "[-180, +180] (360 samples, 1.0 s)", fontsize=9, color="#7D6608")

# Row 3
draw_icon_ecg(550, 1055, size=16, color="#B7950B")
ax.text(580, 1065, "Morphological Vector", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(580, 1047, "x_i ∈ R^{360} Beat Tensor", fontsize=9, color="#7D6608")

draw_icon_trend(770, 1055, size=16, color="#B7950B")
ax.text(805, 1065, "Auxiliary RR Timing (4-d)", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(805, 1047, "Δt_pre, Δt_post, Local_RR, r_pre", fontsize=9, color="#7D6608")

# Row 4
draw_icon_split(550, 960, size=16)
ax.text(580, 972, "DS1 / DS2 Benchmark", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(580, 954, "22 Records Train / 22 Test", fontsize=9, color="#7D6608")

draw_icon_loss(770, 960, size=16, color="#B7950B")
ax.text(805, 972, "Class Weighting", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(805, 954, "w_c = N_total / (5 × N_c)", fontsize=9, color="#7D6608")

# -------------------------------------------------------------------------
# Box 3 (Intermediate, Red/Salmon): Preprocessed Training Data Card
# -------------------------------------------------------------------------
b3_x, b3_y, b3_w, b3_h = 1040, 895, 230, 260
draw_rounded_container(b3_x, b3_y, b3_w, b3_h, "#FDEDEC", "#E74C3C", "Preprocessed\n& Partitioned Data", lw=2.0, title_color="#78281F", title_size=11)
draw_icon_database(b3_x + b3_w/2, 970, size=24, color="#C0392B")
ax.text(b3_x + b3_w/2, 915, "• 42,239 Train Beats\n• 8,756 Val Beats\n• 49,687 Unseen Test", ha='center', va='center', fontsize=9.2, color="#78281F", fontweight='bold')

# -------------------------------------------------------------------------
# Box 4 (Top-Right, Blue): Model Development & Deep Ensemble
# -------------------------------------------------------------------------
b4_x, b4_y, b4_w, b4_h = 1310, 770, 540, 530
draw_rounded_container(b4_x, b4_y, b4_w, b4_h, "#EBF5FB", "#3498DB", "Model Development & Deep Ensemble", lw=2.2, title_color="#154360")

# Sub-items with Icons
draw_icon_network(1350, 1225, size=17, color="#2980B9")
ax.text(1385, 1235, "1D-CNN Feature Extractor Backbone", fontsize=11, fontweight='bold', color="#154360")
ax.text(1385, 1215, "• 3 Blocks: Conv1D(32, 64, 128, k=5) + BN + ReLU + MaxPool(2)", fontsize=9.0, color="#1B4F72")
ax.text(1385, 1197, "• Flattened Waveform Embeddings: 5,760 dimensions", fontsize=9.0, color="#1B4F72")

draw_icon_transfer(1350, 1135, size=16, color="#2980B9")
ax.text(1385, 1145, "Dual-Branch Concatenation & Fusion", fontsize=11, fontweight='bold', color="#154360")
ax.text(1385, 1125, "• Concatenates 5,760 Waveform Dims + 4 RR Timing Dims (5,764 total)", fontsize=9.0, color="#1B4F72")
ax.text(1385, 1107, "• Dense(256) + ReLU + Dropout(p=0.5) + Dense(5 Logits)", fontsize=9.0, color="#1B4F72")

draw_icon_loss(1350, 1045, size=16, color="#2980B9")
ax.text(1385, 1055, "Class-Weighted Cross-Entropy Loss", fontsize=11, fontweight='bold', color="#154360")
ax.text(1385, 1035, "• Severe class rebalancing: w_c = N_total / (5 × N_c)", fontsize=9.0, color="#1B4F72")
ax.text(1385, 1017, "• Support: N (37.8k), S (825), V (3.1k), F (380), Q (8)", fontsize=8.8, color="#1B4F72")

draw_icon_gear(1350, 955, size=16, color="#2980B9")
ax.text(1385, 965, "Adam Optimizer & Regularization", fontsize=11, fontweight='bold', color="#154360")
ax.text(1385, 945, "• Learning rate 1e-3, Batch size 128, Early stopping on val macro F1", fontsize=9.0, color="#1B4F72")

draw_icon_hand(1350, 880, size=16, color="#2980B9")
ax.text(1385, 890, "5-Member Deep Ensemble Architecture", fontsize=11, fontweight='bold', color="#154360")
ax.text(1385, 870, "• 5 Independently Seeded 1D-CNNs (Seeds 0 to 4)", fontsize=9.0, color="#1B4F72")
ax.text(1385, 852, "• Diverse decision boundaries eliminate local minima memorization", fontsize=9.0, color="#1B4F72")

# Top Connecting Arrows (Smooth Left-to-Right)
draw_arrow(b1_x + b1_w, 1035, b2_x, 1035, color="#4CAF50", lw=2.5)
draw_arrow(b2_x + b2_w, 1025, b3_x, 1025, color="#F1C40F", lw=2.5)
draw_arrow(b3_x + b3_w, 1025, b4_x, 1025, color="#E74C3C", lw=2.5)

# =========================================================================
# MIDDLE ROW: VALIDATION CHECK & CLINICAL TRIAGE WORKFLOW
# =========================================================================

# Right Side Flow: Arrow from Model Development down into Validation Check Diamond
diamond_cx, diamond_cy = 1600, 640
draw_arrow(1600, b4_y, 1600, diamond_cy + 55, color="#3498DB", lw=2.5)

# Validation Check Diamond
diamond = Polygon([[diamond_cx, diamond_cy + 55],
                   [diamond_cx + 85, diamond_cy],
                   [diamond_cx, diamond_cy - 55],
                   [diamond_cx - 85, diamond_cy]],
                  closed=True, facecolor="#FDEDEC", edgecolor="#C0392B", linewidth=2.2)
ax.add_patch(diamond)
ax.text(diamond_cx, diamond_cy + 12, "Calibration &", ha='center', va='center', fontsize=10, fontweight='bold', color="#922B21")
ax.text(diamond_cx, diamond_cy - 8, "Val Check", ha='center', va='center', fontsize=10.5, fontweight='bold', color="#922B21")

# NO Branch (Red arrow left into Model Retraining)
draw_arrow(diamond_cx - 85, diamond_cy, 1370, diamond_cy, color="#C0392B", lw=2.2, label="No", label_pos=(1450, diamond_cy + 15))

# YES Branch (Green arrow down into Deployment)
draw_arrow(diamond_cx, diamond_cy - 55, diamond_cx, 440, color="#27AE60", lw=2.2)
draw_arrow(diamond_cx, 440, 1390, 440, color="#27AE60", lw=2.2, label="Yes", label_pos=(1550, 455))

# -------------------------------------------------------------------------
# Middle Purple Container: Active Human-in-the-Loop Clinical Triage
# -------------------------------------------------------------------------
purp_x, purp_y, purp_w, purp_h = 50, 550, 1320, 180
draw_rounded_container(purp_x, purp_y, purp_w, purp_h, "#F4ECF7", "#8E44AD", "Active Human-in-the-Loop Clinical Triage & Continuous Improvement", lw=2.0, title_color="#512E5F")

# 4 Sequential Steps inside Purple Container
step_w = 270
s1_x = 75
s2_x = 390
s3_x = 710
s4_x = 1030
sy = 570
sh = 115

# Step 1: Case Selection
c1 = FancyBboxPatch((s1_x, sy), step_w, sh, boxstyle="round,pad=0,rounding_size=8", facecolor="#FFFFFF", edgecolor="#D2B4DE", lw=1.2)
ax.add_patch(c1)
ax.text(s1_x + 10, sy + 95, "Case Selection", fontsize=10.5, fontweight='bold', color="#512E5F")
ax.text(s1_x + 10, sy + 75, "• Select high uncertainty beats", fontsize=8.8, color="#512E5F")
ax.text(s1_x + 10, sy + 55, "• Threshold u(x) > τ (10%–30%)", fontsize=8.8, color="#512E5F")
ax.text(s1_x + 10, sy + 35, "• Catches misclassified ectopics", fontsize=8.8, color="#512E5F")
ax.text(s1_x + 10, sy + 15, "• Flags ambiguous morphologies", fontsize=8.8, color="#512E5F")

draw_arrow(s1_x + step_w, sy + sh/2, s2_x, sy + sh/2, color="#8E44AD", lw=2.0)

# Step 2: Expert Review
c2 = FancyBboxPatch((s2_x, sy), step_w, sh, boxstyle="round,pad=0,rounding_size=8", facecolor="#FFFFFF", edgecolor="#D2B4DE", lw=1.2)
ax.add_patch(c2)
ax.text(s2_x + 10, sy + 95, "Expert Cardiologist Review", fontsize=10.5, fontweight='bold', color="#512E5F")
ax.text(s2_x + 10, sy + 75, "• Cardiologists review flagged beats", fontsize=8.8, color="#512E5F")
ax.text(s2_x + 10, sy + 55, "• Resolves Supraventricular (S)", fontsize=8.8, color="#512E5F")
ax.text(s2_x + 10, sy + 35, "• Resolves Ventricular Fusion (F)", fontsize=8.8, color="#512E5F")
ax.text(s2_x + 10, sy + 15, "• Validates rhythm plausibility", fontsize=8.8, color="#512E5F")

draw_arrow(s2_x + step_w, sy + sh/2, s3_x, sy + sh/2, color="#8E44AD", lw=2.0)

# Step 3: Dataset Refinement
c3 = FancyBboxPatch((s3_x, sy), step_w, sh, boxstyle="round,pad=0,rounding_size=8", facecolor="#FFFFFF", edgecolor="#D2B4DE", lw=1.2)
ax.add_patch(c3)
ax.text(s3_x + 10, sy + 95, "Clinical EMR Logging", fontsize=10.5, fontweight='bold', color="#512E5F")
ax.text(s3_x + 10, sy + 75, "• Verified diagnoses stored in EMR", fontsize=8.8, color="#512E5F")
ax.text(s3_x + 10, sy + 55, "• Patient rhythm history updated", fontsize=8.8, color="#512E5F")
ax.text(s3_x + 10, sy + 35, "• Audit trail for medical safety", fontsize=8.8, color="#512E5F")
ax.text(s3_x + 10, sy + 15, "• High-priority clinical alerts", fontsize=8.8, color="#512E5F")

draw_arrow(s3_x + step_w, sy + sh/2, s4_x, sy + sh/2, color="#8E44AD", lw=2.0)

# Step 4: Model Retraining
c4 = FancyBboxPatch((s4_x, sy), step_w, sh, boxstyle="round,pad=0,rounding_size=8", facecolor="#FFFFFF", edgecolor="#D2B4DE", lw=1.2)
ax.add_patch(c4)
ax.text(s4_x + 10, sy + 95, "Patient-Specific Adaptation", fontsize=10.5, fontweight='bold', color="#512E5F")
ax.text(s4_x + 10, sy + 75, "• Fine-tunes model on verified cases", fontsize=8.8, color="#512E5F")
ax.text(s4_x + 10, sy + 55, "• Bridges inter-patient gap", fontsize=8.8, color="#512E5F")
ax.text(s4_x + 10, sy + 35, "• Domain-adapted Holter telemetry", fontsize=8.8, color="#512E5F")
ax.text(s4_x + 10, sy + 15, "• Continuous active learning", fontsize=8.8, color="#512E5F")

# -------------------------------------------------------------------------
# Middle Row: Deployment, Operational Predictions, Low-Confidence Cases
# (Right-to-Left Flow, exactly like reference image)
# -------------------------------------------------------------------------
m_y = 370
m_h = 135

# Right: Deployment (Green)
dep_x = 1170
dep_w = 220
draw_rounded_container(dep_x, m_y, dep_w, m_h, "#EAF8E6", "#27AE60", "Deployment", lw=2.0, title_color="#1E8449", title_size=11.5)
draw_icon_phone(dep_x + 40, m_y + 55, size=18, color="#27AE60")
ax.text(dep_x + 80, m_y + 75, "• Ambulatory Holter", fontsize=9.2, color="#1E8449")
ax.text(dep_x + 80, m_y + 55, "• Real-Time Monitor", fontsize=9.2, color="#1E8449")
ax.text(dep_x + 80, m_y + 35, "• Hospital Telemetry", fontsize=9.2, color="#1E8449")

# Arrow from Deployment to Operational Predictions
draw_arrow(dep_x, m_y + m_h/2, 1090, m_y + m_h/2, color="#27AE60", lw=2.2)

# Center: Operational Predictions (Yellow)
op_x = 680
op_w = 410
draw_rounded_container(op_x, m_y, op_w, m_h, "#FEFDE8", "#F1C40F", "Operational Predictions", lw=2.0, title_color="#7D6608", title_size=11.5)
draw_icon_trend(op_x + 40, m_y + 55, size=18, color="#F39C12")
ax.text(op_x + 80, m_y + 80, "• Highly Accurate: 89.60% (10% ref) to 94.70% (30% ref)", fontsize=9.2, fontweight='bold', color="#7D6608")
ax.text(op_x + 80, m_y + 60, "• Trustworthy: Retained beats have high certainty u(x) ≤ τ", fontsize=9.2, color="#7D6608")
ax.text(op_x + 80, m_y + 40, "• Autonomous EMR logging for Normal (N) & Ventricular (V)", fontsize=9.2, color="#7D6608")
ax.text(op_x + 80, m_y + 20, "• Reduces cardiologist review burden by 70% to 90%", fontsize=9.2, color="#7D6608")

# Arrow from Operational Predictions to Low-Confidence Cases
draw_arrow(op_x, m_y + m_h/2, 600, m_y + m_h/2, color="#E67E22", lw=2.2)

# Left: Low confidence / High uncertainty cases (Orange/Peach)
lc_x = 150
lc_w = 450
draw_rounded_container(lc_x, m_y, lc_w, m_h, "#FDF2E9", "#E67E22", "Low Confidence / High Uncertainty Cases", lw=2.0, title_color="#A04000", title_size=11.5)
draw_icon_triage(lc_x + 40, m_y + 55, size=18, color="#E67E22")
ax.text(lc_x + 80, m_y + 80, "• Ambiguous morphology or high entropy u(x) > τ", fontsize=9.2, fontweight='bold', color="#A04000")
ax.text(lc_x + 80, m_y + 60, "• Intercepts difficult Supraventricular (S) & Fusion (F) beats", fontsize=9.2, color="#A04000")
ax.text(lc_x + 80, m_y + 40, "• Protects against catastrophic diagnostic false negatives", fontsize=9.2, color="#A04000")
ax.text(lc_x + 80, m_y + 20, "• Automatically diverted for manual clinical review", fontsize=9.2, color="#A04000")

# Arrow from Low confidence cases UP into Case Selection in Purple Container
draw_arrow(lc_x + lc_w/2, m_y + m_h, s1_x + step_w/2, sy, color="#E67E22", lw=2.4)

# =========================================================================
# BOTTOM ROW: MODEL PREDICTIONS, UQ EXPLANATIONS, AND DOMAIN VERIFICATION
# =========================================================================
bot_y = 60
bot_h = 240

# Bottom Left (Yellow): Model Predictions (Class Probabilities)
p1_x = 50
p1_w = 430
draw_rounded_container(p1_x, bot_y, p1_w, bot_h, "#FEF9E7", "#F1C40F", "Model Predictions (Class Probabilities)", lw=2.0, title_color="#7D6608", title_size=11.5)
draw_icon_network(p1_x + 40, bot_y + 160, size=18, color="#D4AC0D")
ax.text(p1_x + 75, bot_y + 195, "Calibrated Ensemble Posteriors", fontsize=10.5, fontweight='bold', color="#7D6608")
ax.text(p1_x + 75, bot_y + 175, "• Member Logits: z_m ∈ R^5 (m = 1...5)", fontsize=9.0, color="#7D6608")
ax.text(p1_x + 75, bot_y + 155, "• Temperature Scaling: T = [0.84, 1.18, 1.49, 0.98, 1.33]", fontsize=9.0, color="#7D6608")
ax.text(p1_x + 75, bot_y + 135, "• Logit Averaging (Eq. 5): z_avg = (1/M) Σ (z_m / T_m)", fontsize=9.0, color="#7D6608")
ax.text(p1_x + 75, bot_y + 115, "• Final Posterior: p_ens = Softmax(z_avg)", fontsize=9.0, color="#7D6608")
ax.text(p1_x + 75, bot_y + 95, "• Unseen Test Accuracy: 85.86% (DS2, 49,687 beats)", fontsize=9.0, color="#7D6608")
ax.text(p1_x + 75, bot_y + 75, "• Macro F1: 0.371 (5-Class) / 0.463 (4-Class)", fontsize=9.0, color="#7D6608")

# Arrow from Model Predictions to UQ Explanations
draw_arrow(p1_x + p1_w, bot_y + bot_h/2, 515, bot_y + bot_h/2, color="#F1C40F", lw=2.2)

# Bottom Center (Blue): Post-hoc Uncertainty Quantification
p2_x = 515
p2_w = 640
draw_rounded_container(p2_x, bot_y, p2_w, bot_h, "#EBF5FB", "#3498DB", "Post-hoc Uncertainty Quantification (Risk Attributions)", lw=2.0, title_color="#154360", title_size=11.5)
draw_icon_filter(p2_x + 40, bot_y + 160, size=18, color="#2980B9")
ax.text(p2_x + 75, bot_y + 195, "Benchmark of 4 Clinical Uncertainty Formulations", fontsize=10.5, fontweight='bold', color="#154360")
ax.text(p2_x + 75, bot_y + 175, "1. Softmax Confidence (SC): u_SC = 1 - max_c p_ens(c|x)", fontsize=9.0, fontweight='bold', color="#1B4F72")
ax.text(p2_x + 95, bot_y + 157, "• Superior Error Detector | AUROC: 0.819 | AUPRC: 0.405 | AURC: 0.0465", fontsize=8.6, color="#1B4F72")
ax.text(p2_x + 75, bot_y + 138, "2. Predictive Entropy (PE): u_PE = -Σ p_ens log2(p_ens) [AUROC: 0.810]", fontsize=9.0, color="#1B4F72")
ax.text(p2_x + 75, bot_y + 119, "3. Cluster-Based Entropy (CBE): Groups into 4 tiers {N}, {S}, {V,F}, {Q}", fontsize=9.0, color="#1B4F72")
ax.text(p2_x + 95, bot_y + 101, "• AUROC: 0.798 | Captures risk transitions without extra latency", fontsize=8.6, color="#1B4F72")
ax.text(p2_x + 75, bot_y + 82, "4. MC-Dropout Variance (MCD): Epistemic dispersion over 20 passes [AUROC: 0.754]", fontsize=8.8, color="#1B4F72")

# Arrow from UQ Explanations to Domain Verification
draw_arrow(p2_x + p2_w, bot_y + bot_h/2, 1190, bot_y + bot_h/2, color="#3498DB", lw=2.2)

# Bottom Right (Purple): Clinical Validation & Cardiology Domain Verification
p3_x = 1190
p3_w = 660
draw_rounded_container(p3_x, bot_y, p3_w, bot_h, "#F5EEF8", "#8E44AD", "Clinical Validation & Cardiology Domain Verification", lw=2.0, title_color="#512E5F", title_size=11.5)
draw_icon_doctor(p3_x + 40, bot_y + 160, size=18, color="#8E44AD")
ax.text(p3_x + 75, bot_y + 195, "Comprehensive Diagnostic & Reliability Metrics", fontsize=10.5, fontweight='bold', color="#512E5F")
ax.text(p3_x + 75, bot_y + 175, "• Calibration Fidelity: ECE reduced by 29% (4.28% -> 3.04%), NLL: 0.4020", fontsize=9.0, color="#512E5F")
ax.text(p3_x + 75, bot_y + 155, "• Risk-Coverage Analysis (AURC): 0.0465 under Softmax Confidence", fontsize=9.0, color="#512E5F")
ax.text(p3_x + 75, bot_y + 135, "• Selective Triage Accuracy Sweep on 49,687 Unseen Test Beats (DS2):", fontsize=9.0, fontweight='bold', color="#512E5F")
ax.text(p3_x + 95, bot_y + 115, "• 0% Referral: 85.86% Acc  |  Macro F1: 0.371", fontsize=8.8, color="#512E5F")
ax.text(p3_x + 95, bot_y + 95, "• 10% Referral: 89.60% Acc  |  Macro F1: 0.390 (Optimal Triage)", fontsize=8.8, fontweight='bold', color="#512E5F")
ax.text(p3_x + 95, bot_y + 75, "• 30% Referral: 94.70% Acc  |  Macro F1: 0.420 (High Precision Mode)", fontsize=8.8, color="#512E5F")

# Orthogonal Return Arrow from Clinical Verification back up to Validation Check Diamond
ret_x = 1865
ax.plot([p3_x + p3_w - 10, ret_x], [bot_y + bot_h/2, bot_y + bot_h/2], color="#C0392B", lw=2.0)
ax.plot([ret_x, ret_x], [bot_y + bot_h/2, diamond_cy], color="#C0392B", lw=2.0)
draw_arrow(ret_x, diamond_cy, diamond_cx + 85, diamond_cy, color="#C0392B", lw=2.0)

# Save figures
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
print("Replica architecture diagram generated successfully!")
