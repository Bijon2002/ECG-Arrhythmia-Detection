import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, lfilter

output_dir = r'C:\Users\BIJON\.gemini\antigravity-ide\brain\65640336-4456-4dd3-a96a-17ced2738e84'

def generate_5_class_ecg():
    t = np.linspace(0, 1, 360)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    n_beat = np.sin(2*np.pi*1*t)*0.1 + np.exp(-((t-0.5)/0.02)**2) - np.exp(-((t-0.55)/0.02)**2)*0.3
    s_beat = np.sin(2*np.pi*1*t)*0.1 + np.exp(-((t-0.5)/0.02)**2) + np.exp(-((t-0.35)/0.03)**2)*0.5
    v_beat = np.exp(-((t-0.5)/0.08)**2)*1.2 - np.exp(-((t-0.6)/0.08)**2)*0.8
    f_beat = n_beat * 0.5 + v_beat * 0.5
    q_beat = n_beat.copy()
    q_beat[150:152] = 2.0
    
    beats = [n_beat, s_beat, v_beat, f_beat, q_beat]
    titles = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Paced (Q)']
    
    for ax, beat, title in zip(axes, beats, titles):
        ax.plot(t, beat, color='#1f77b4')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_1_class_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_uncertainty_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    t = np.linspace(0, 10, 100)
    y_true = np.sin(t)
    y_noisy = y_true + np.random.normal(0, 0.5, 100)
    ax1.scatter(t, y_noisy, color='gray', alpha=0.5, label='Noisy Data')
    ax1.plot(t, y_true, color='blue', label='True Signal')
    ax1.fill_between(t, y_true-0.5, y_true+0.5, color='blue', alpha=0.2, label='Aleatoric Uncertainty')
    ax1.set_title('Aleatoric Uncertainty\n(Noise inherent in the ECG signal)')
    ax1.legend(loc='lower left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    t_train = np.linspace(0, 4, 40)
    y_train = np.sin(t_train)
    t_test = np.linspace(0, 10, 100)
    
    ax2.scatter(t_train, y_train, color='black', label='Training Data')
    ax2.plot(t_test, np.sin(t_test), color='red', linestyle='--', label='Mean Prediction')
    
    env = np.zeros_like(t_test)
    env[t_test > 4] = (t_test[t_test > 4] - 4) * 0.4
    ax2.fill_between(t_test, np.sin(t_test)-env, np.sin(t_test)+env, color='red', alpha=0.2, label='Epistemic Uncertainty')
    
    ax2.set_title('Epistemic Uncertainty\n(Model lacks knowledge of rare arrhythmias)')
    ax2.legend(loc='lower left')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_2_epistemic_aleatoric.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_filter_diagram():
    t = np.linspace(0, 2, 720)
    y_true = np.sin(2*np.pi*1.2*t) + np.exp(-((t%1 - 0.5)/0.02)**2)*2
    y_noisy = y_true + np.sin(2*np.pi*0.2*t)*1.5 + np.random.normal(0, 0.3, 720)
    
    b, a = butter(2, [0.5/(360/2), 45/(360/2)], btype='bandpass')
    y_filtered = lfilter(b, a, y_noisy)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(t, y_noisy, color='gray')
    ax1.set_title('Raw ECG Signal (with Baseline Wander and High-Frequency Noise)', fontsize=12)
    ax1.axis('off')
    
    ax2.plot(t, y_filtered, color='#1f77b4')
    ax2.set_title('Filtered ECG Signal (0.5-45 Hz Butterworth Band-Pass)', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_3_filter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_mcdropout_diagram():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.linspace(0, 1, 100)
    for i in range(10):
        y = np.exp(-((t-0.5)/0.1)**2) * (1 + np.random.normal(0, 0.1))
        ax.plot(t, y, color='red', alpha=0.3)
    
    ax.plot(t, np.exp(-((t-0.5)/0.1)**2), color='darkred', linewidth=3, label='Mean Prediction')
    
    ax.set_title('Monte Carlo Dropout Inference\n(10 Stochastic Forward Passes)', fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    ax.text(0.05, 0.8, 'Input: Ambiguous Beat', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.05, 0.6, 'Output: Prediction Distribution\nVariance = Epistemic Uncertainty', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_4_mcdropout.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_calibration_diagram():
    fig, ax = plt.subplots(figsize=(6, 6))
    
    conf = np.linspace(0, 1, 10)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    acc_pre = conf * 0.7 + np.random.normal(0, 0.05, 10)
    acc_pre = np.clip(acc_pre, 0, 1)
    ax.plot(conf, acc_pre, 'r-o', label='Pre-Calibration (Overconfident)')
    
    acc_post = conf * 0.95 + np.random.normal(0, 0.02, 10)
    acc_post = np.clip(acc_post, 0, 1)
    ax.plot(conf, acc_post, 'g-s', label='Post-Temperature Scaling (Calibrated)')
    
    ax.set_xlabel('Model Confidence', fontsize=12)
    ax.set_ylabel('Empirical Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram (Expected Calibration Error)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_5_calibration.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_5_class_ecg()
    generate_uncertainty_diagram()
    generate_filter_diagram()
    generate_mcdropout_diagram()
    generate_calibration_diagram()
    print("Images generated successfully.")
