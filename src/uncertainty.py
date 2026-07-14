import numpy as np

def calculate_predictive_entropy(mean_probs):
    """
    Standard Shannon Entropy formula across all 5 classes.
    Formula: -sum(p * log(p))
    """
    epsilon = 1e-10 # Prevent log(0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon))
    return float(entropy)

def calculate_cluster_based_entropy(mean_probs):
    """
    Cluster-based entropy incorporates medical domain knowledge as specified in the report.
    It groups the 5 classes into 4 clinically meaningful clusters:
    Cluster 0: Normal Rhythms (Class 0)
    Cluster 1: Atrial Problems (Class 1)
    Cluster 2: Ventricular Issues (Classes 2 and 3)
    Cluster 3: Unknown / Other (Class 4)
    """
    cluster_probs = np.array([
        mean_probs[0],                 # Normal
        mean_probs[1],                 # Supraventricular (Atrial)
        mean_probs[2] + mean_probs[3], # Ventricular + Fusion
        mean_probs[4]                  # Unknown
    ])
    
    epsilon = 1e-10
    cluster_entropy = -np.sum(cluster_probs * np.log(cluster_probs + epsilon))
    return float(cluster_entropy)
