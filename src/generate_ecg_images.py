import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    data = np.load('f:/ECG-Arrhythmia-Detection/data/processed/test_data.npz')
    return data['X'], data['y']

def setup_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#080c16',
        'axes.facecolor': '#080c16',
        'axes.edgecolor': '#1f2937',
        'axes.grid': True,
        'grid.color': '#1f2937',
        'grid.alpha': 0.5,
        'xtick.color': '#6b7280',
        'ytick.color': '#6b7280',
    })

def plot_class(signal, color, title, filename):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(signal, color=color, linewidth=2)
    # ax.set_title(title, color='#ffffff', pad=10) # We will rely on HTML for title
    ax.axis('off') # Cleaner look for the cards
    plt.tight_layout()
    plt.savefig(f'f:/ECG-Arrhythmia-Detection/frontend/{filename}', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def plot_anatomy(signal, color, filename):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal, color=color, linewidth=2)
    
    # We need to guess the peaks for annotation
    # Usually R peak is the max
    r_idx = np.argmax(signal)
    
    # Hardcode offsets for a typical Normal beat
    p_idx = r_idx - 40
    t_idx = r_idx + 55
    
    # Ensure indices are within bounds
    p_idx = max(0, p_idx)
    t_idx = min(len(signal)-1, t_idx)

    # Expand Y limits so text has room
    ax.set_ylim(min(signal) - 0.2, max(signal) + 0.9)

    # Annotate P wave (Move text up and left)
    ax.annotate('P-Wave\n(Atrial Contraction)', 
                xy=(p_idx, signal[p_idx]), 
                xytext=(p_idx - 25, signal[p_idx] + 0.4),
                arrowprops=dict(facecolor='white', arrowstyle='->', color='white'),
                color='white', ha='center', fontsize=10)
                
    # Annotate QRS (Move text straight up above peak)
    ax.annotate('QRS Complex\n(Ventricular Contraction)', 
                xy=(r_idx, signal[r_idx]), 
                xytext=(r_idx, signal[r_idx] + 0.4),
                arrowprops=dict(facecolor='#00ff88', arrowstyle='->', color='#00ff88', lw=2),
                color='#00ff88', ha='center', fontsize=11, fontweight='bold')
                
    # Annotate T wave (Move text up and right)
    ax.annotate('T-Wave\n(Reset)', 
                xy=(t_idx, signal[t_idx]), 
                xytext=(t_idx + 30, signal[t_idx] + 0.4),
                arrowprops=dict(facecolor='white', arrowstyle='->', color='white'),
                color='white', ha='center', fontsize=10)
                
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'f:/ECG-Arrhythmia-Detection/frontend/{filename}', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def main():
    X, y = load_data()
    setup_plot_style()
    
    # Find one example of each class
    # Classes: 0:N, 1:S, 2:V, 3:F, 4:Q
    classes = [0, 1, 2, 3, 4]
    colors = ['#00ff88', '#ffb703', '#ff6b6b', '#a855f7', '#9ca3af']
    titles = ['Normal (N)', 'Supraventricular Ectopic (S)', 'Ventricular Ectopic (V)', 'Fusion (F)', 'Unknown (Q)']
    filenames = ['ecg_class_n.png', 'ecg_class_s.png', 'ecg_class_v.png', 'ecg_class_f.png', 'ecg_class_q.png']
    
    examples = {}
    for c in classes:
        idx = np.where(y == c)[0][0] # first example
        # Avoid extreme outliers, pick a decent looking one (e.g., 5th example)
        idx = np.where(y == c)[0][4] if len(np.where(y == c)[0]) > 4 else idx
        examples[c] = X[idx].flatten()
        
    for c in classes:
        plot_class(examples[c], colors[c], titles[c], filenames[c])
        print(f"Saved {filenames[c]}")
        
    # Plot anatomy using the Normal beat
    plot_anatomy(examples[0], '#00b4d8', 'ecg_anatomy.png')
    print("Saved ecg_anatomy.png")

if __name__ == '__main__':
    main()
