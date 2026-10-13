import wfdb
import os

base_dir = r'f:\ECG-Arrhythmia-Detection\data\raw'
recs = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230']

for r in sorted(recs):
    h = wfdb.rdheader(os.path.join(base_dir, r))
    comments = " ".join(h.comments).strip() if h.comments else "None"
    print(f"{r}: {comments}")
