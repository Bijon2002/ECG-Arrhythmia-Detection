import numpy as np
import pandas as pd
import os

# Connect to our data folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data', 'samples', 'normal_sample.csv')

def fix_dummy_sample():
    print("Loading datasets...")
    test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
    X = test_data['X']
    y = test_data['y']

    # Find the very first Normal heartbeat in our test set
    normal_idx = np.where(y == 0)[0][0]
    normal_beat = X[normal_idx]

    # Save it as a 1-column CSV file directly over the old dummy file
    pd.DataFrame(normal_beat).to_csv(SAMPLE_PATH, index=False, header=False)
    print(f"✅ Successfully extracted a real 360-sample heartbeat and saved it to: {SAMPLE_PATH}")

if __name__ == "__main__":
    fix_dummy_sample()
