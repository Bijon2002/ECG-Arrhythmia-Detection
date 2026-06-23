import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# We use absolute paths to ensure the script works from anywhere, 
# but for the project we'll use relative paths from the root directory.
DATA_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
FS = 360  # Sampling frequency
WINDOW_SIZE = 180  # 180 samples left and right (0.5s each side)

# AAMI classes mapping
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular
    'V': 2, 'E': 2,                          # Ventricular
    'F': 3,                                  # Fusion
    '/': 4, 'f': 4, 'Q': 4                   # Unknown
}

def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_records():
    """Download records if necessary, filter, and extract beats."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Fetching record list for MIT-BIH dataset...")
    record_list = wfdb.get_record_list('mitdb')
    
    # Download the records if they don't exist
    if not os.path.exists(os.path.join(DATA_DIR, '100.dat')):
        print("Downloading MIT-BIH database to data/raw/ (this may take a moment)...")
        wfdb.dl_database('mitdb', dl_dir=DATA_DIR)
    else:
        print("Dataset already downloaded in data/raw/!")
        
    all_beats = []
    all_labels = []
    
    print("Processing 48 patient records (Filtering & Slicing)...")
    for record_name in tqdm(record_list):
        record_path = os.path.join(DATA_DIR, record_name)
        
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except FileNotFoundError:
            continue
            
        signal = record.p_signal[:, 0]
        filtered_signal = apply_bandpass_filter(signal)
        
        peaks = annotation.sample
        symbols = annotation.symbol
        
        # Loop through each peak
        for peak, symbol in zip(peaks, symbols):
            # Check if this heartbeat's symbol is one of the ones we want to learn
            if symbol in AAMI_MAPPING:
                # Ensure we have enough data to the left and right
                if peak - WINDOW_SIZE >= 0 and peak + WINDOW_SIZE < len(filtered_signal):
                    beat = filtered_signal[peak - WINDOW_SIZE : peak + WINDOW_SIZE]
                    label = AAMI_MAPPING[symbol]
                    
                    all_beats.append(beat)
                    all_labels.append(label)
                    
    print(f"\nExtraction complete! Successfully extracted {len(all_beats)} individual heartbeats.")
    return np.array(all_beats), np.array(all_labels)

if __name__ == "__main__":
    X, y = process_records()
    print(f"\nFinal Data Matrix Shape (X): {X.shape}")
    print(f"Final Labels Shape (y): {y.shape}")
    
    print("\nSplitting data into 80% Training and 20% Testing...")
    # stratify=y ensures both sets have the same ratio of arrhythmias
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Set: {X_train.shape[0]} heartbeats")
    print(f"Testing Set: {X_test.shape[0]} heartbeats")
    
    print("\nSaving highly compressed data to data/processed/...")
    np.savez_compressed(os.path.join(PROCESSED_DIR, 'train_data.npz'), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(PROCESSED_DIR, 'test_data.npz'), X=X_test, y=y_test)
    print("Data saved successfully! Phase 2 is complete.")
