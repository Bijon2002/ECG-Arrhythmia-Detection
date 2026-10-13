import os
import wfdb
import json
import numpy as np
from collections import Counter
from scipy.signal import butter, filtfilt
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'inter_patient')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# de Chazal et al. (2004) DS1 & DS2 splits (excluding paced records 102, 104, 107, 217)
DS1_TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119', '122',
    '201', '203', '207', '208', '209', '215', '220', '230'
]
DS1_VAL_RECORDS = ['114', '124', '205', '223']

DS2_TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221', '222', '228',
    '231', '232', '233', '234'
]

# AAMI 4-class mapping
AAMI_4CLASS = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic (S)
    'V': 2, 'E': 2,                          # Ventricular ectopic (V)
    'F': 3                                   # Fusion (F)
}

HEARTBEAT_SYMBOLS = set(AAMI_4CLASS.keys()).union({'B', 'r', 'n', 'E', '/', 'f', 'Q', '?'})
CLASS_NAMES = ['N', 'S', 'V', 'F']
FS = 360.0
WINDOW_SIZE = 180  # 180 before, 180 after = 360 samples (+-0.5s)

def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_record_list(record_list, split_name):
    print(f"\nProcessing {split_name} ({len(record_list)} records)...")
    beats = []
    rr_feats = []
    labels = []
    record_ids = []
    per_record_stats = {}

    for record_name in tqdm(record_list):
        record_path = os.path.join(DATA_DIR, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        # Check lead name: select MLII strictly by name
        if 'MLII' in record.sig_name:
            lead_idx = record.sig_name.index('MLII')
        else:
            lead_idx = 0
            print(f"Warning: MLII not found in record {record_name}, using lead index 0 ({record.sig_name[0]})")

        raw_signal = record.p_signal[:, lead_idx]
        filtered_signal = apply_bandpass_filter(raw_signal, lowcut=0.5, highcut=45.0, fs=FS, order=2)

        # Filter strictly to heartbeat annotations for accurate RR calculation
        beat_indices = [i for i, s in enumerate(annotation.symbol) if s in HEARTBEAT_SYMBOLS]
        peaks = [annotation.sample[i] for i in beat_indices]
        symbols = [annotation.symbol[i] for i in beat_indices]

        # Compute per-record median RR interval for robust normalization
        rr_raw = np.diff(peaks) / FS
        med_rr = np.median(rr_raw) if len(rr_raw) > 0 else 0.8

        rec_counts = Counter()

        for idx, (peak, symbol) in enumerate(zip(peaks, symbols)):
            if symbol in AAMI_4CLASS:
                if peak - WINDOW_SIZE >= 0 and peak + WINDOW_SIZE < len(filtered_signal):
                    # 1. Waveform slice
                    beat = filtered_signal[peak - WINDOW_SIZE : peak + WINDOW_SIZE]
                    beat_std = np.std(beat)
                    beat_norm = (beat - np.mean(beat)) / (beat_std + 1e-8)

                    # 2. De Chazal RR Timing Features
                    pre_rr = (peaks[idx] - peaks[idx - 1]) / FS if idx > 0 else med_rr
                    post_rr = (peaks[idx + 1] - peaks[idx]) / FS if idx < len(peaks) - 1 else med_rr

                    start_idx = max(0, idx - 5)
                    end_idx = min(len(peaks), idx + 6)
                    if end_idx > start_idx + 1:
                        local_rr = (peaks[end_idx - 1] - peaks[start_idx]) / (FS * (end_idx - start_idx - 1))
                    else:
                        local_rr = med_rr

                    feat_pre = pre_rr / (med_rr + 1e-5)
                    feat_post = post_rr / (med_rr + 1e-5)
                    feat_ratio = pre_rr / (local_rr + 1e-5)

                    label = AAMI_4CLASS[symbol]
                    beats.append(beat_norm)
                    rr_feats.append([feat_pre, feat_post, feat_ratio])
                    labels.append(label)
                    record_ids.append(record_name)
                    rec_counts[CLASS_NAMES[label]] += 1

        per_record_stats[record_name] = dict(rec_counts)

    beats = np.array(beats, dtype=np.float32)
    rr_feats = np.array(rr_feats, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    record_ids = np.array(record_ids)

    return beats, rr_feats, labels, record_ids, per_record_stats

def main():
    print("=" * 60)
    print("Inter-Patient Dataset Extraction (de Chazal DS1/DS2 - 4 Classes + RR)")
    print("=" * 60)

    # 1. Process Train
    X_train, RR_train, y_train, rec_train, stats_train = process_record_list(DS1_TRAIN_RECORDS, "DS1-Train")
    # 2. Process Val
    X_val, RR_val, y_val, rec_val, stats_val = process_record_list(DS1_VAL_RECORDS, "DS1-Val")
    # 3. Process Test
    X_test, RR_test, y_test, rec_test, stats_test = process_record_list(DS2_TEST_RECORDS, "DS2-Test")

    # Save to compressed npz
    print("\nSaving processed datasets to data/inter_patient/ ...")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'train_data.npz'),
        X=X_train, RR=RR_train, y=y_train, record_ids=rec_train
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'val_data.npz'),
        X=X_val, RR=RR_val, y=y_val, record_ids=rec_val
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'test_data.npz'),
        X=X_test, RR=RR_test, y=y_test, record_ids=rec_test
    )

    all_stats = {
        'train': {
            'records': DS1_TRAIN_RECORDS,
            'total': len(y_train),
            'class_counts': {CLASS_NAMES[i]: int(np.sum(y_train == i)) for i in range(4)},
            'per_record': stats_train
        },
        'val': {
            'records': DS1_VAL_RECORDS,
            'total': len(y_val),
            'class_counts': {CLASS_NAMES[i]: int(np.sum(y_val == i)) for i in range(4)},
            'per_record': stats_val
        },
        'test': {
            'records': DS2_TEST_RECORDS,
            'total': len(y_test),
            'class_counts': {CLASS_NAMES[i]: int(np.sum(y_test == i)) for i in range(4)},
            'per_record': stats_test
        }
    }

    stats_path = os.path.join(RESULTS_DIR, 'inter_patient_data_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("INTER-PATIENT DATASET SUMMARY (Morphology + RR)")
    print("=" * 60)
    for split, data in all_stats.items():
        print(f"\n{split.upper()} (Total: {data['total']:,} beats across {len(data['records'])} patients):")
        for cls_name, cnt in data['class_counts'].items():
            print(f"  Class {cls_name}: {cnt:>6,} ({cnt/data['total']*100:>5.2f}%)")

    print(f"\nStats written to: {stats_path}")
    print("Preprocessing completed successfully!")

if __name__ == '__main__':
    main()
