import matplotlib.pyplot as plt
import os
import seaborn as sns

def generate_distribution_plot():
    classes = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']
    counts = [18118, 556, 1448, 162, 1604]
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create bar plot
    ax = sns.barplot(x=classes, y=counts, palette="viridis")
    
    # Add labels on top of bars
    for i, v in enumerate(counts):
        ax.text(i, v + 200, str(v), ha='center', va='bottom', fontweight='bold')
        
    plt.title('MIT-BIH Arrhythmia Dataset - Class Distribution (Test Set)', fontsize=14, pad=15)
    plt.ylabel('Number of Heartbeats', fontsize=12)
    plt.xlabel('AAMI Class', fontsize=12)
    
    # Adjust y-axis to make room for labels
    plt.ylim(0, 20000)
    
    # Save the plot
    os.makedirs('f:/ECG-Arrhythmia-Detection/results/plots', exist_ok=True)
    save_path = 'f:/ECG-Arrhythmia-Detection/results/plots/dataset_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated dataset distribution plot at: {save_path}")

if __name__ == "__main__":
    generate_distribution_plot()
