import os
import glob
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "weighted_rr")

print("=" * 65)
print("     ECG Arrhythmia AI — Live Training Status & Epoch Tracker")
print("=" * 65)

# Check saved models
saved_models = sorted(glob.glob(os.path.join(MODELS_DIR, "run0_m*.pt")))
saved_jsons = sorted(glob.glob(os.path.join(MODELS_DIR, "run0_m*.json")))

print(f"\nModels Completed: {len(saved_models)} / 5")
for i in range(5):
    pt_file = os.path.join(MODELS_DIR, f"run0_m{i}.pt")
    if os.path.exists(pt_file):
        size_mb = os.path.getsize(pt_file) / (1024 * 1024)
        print(f"  [COMPLETED] Member {i+1}/5: run0_m{i}.pt ({size_mb:.2f} MB)")
    else:
        if i == len(saved_models):
            print(f"  [TRAINING NOW...] Member {i+1}/5: run0_m{i}.pt")
        else:
            print(f"  [PENDING]         Member {i+1}/5: run0_m{i}.pt")

# Look for latest live training log output
log_pattern = r"C:\Users\BIJON\.gemini\antigravity-ide\brain\*\.system_generated\tasks\task-*.log"
log_files = glob.glob(log_pattern)
if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    try:
        with open(latest_log, "r", encoding="utf-8", errors="ignore") as lf:
            lines = [l.strip() for l in lf.readlines() if l.strip()]
            epoch_lines = [l for l in lines if "Epoch" in l or "Training Member" in l or "Early stopping" in l]
            if epoch_lines:
                print("\n" + "-" * 65)
                print("LATEST TRAINING PROGRESS (Live Epoch Logs):")
                for el in epoch_lines[-5:]:
                    print(f"  {el}")
    except Exception:
        pass

print("\n" + "=" * 65)
if len(saved_models) == 5:
    print("🎉 ALL 5 ENSEMBLE MODELS ARE TRAINED AND READY!")
    print("\nYou can now run the evaluation command:")
    print("  python pipeline_v2/evaluate.py --run 0 --weighted 1 --rr 1 --split interpatient --mc 10")
else:
    print(f"Training in progress ({len(saved_models)} of 5 models completed).")
    print("Estimated time remaining: ~5 to 8 minutes on CPU.")
    print("Run 'python check_status.py' anytime to refresh.")
print("=" * 65)
