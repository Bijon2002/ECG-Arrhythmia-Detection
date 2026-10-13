import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModel

DATA_DIR = os.path.join(BASE_DIR, 'data', 'inter_patient')
MODELS_DIR = os.path.join(BASE_DIR, 'models_inter')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs_inter')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics_inter')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

CLASS_NAMES = ['N', 'S', 'V', 'F']
SEEDS = [42, 101, 202, 303, 404]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(batch_size=256):
    train_npz = np.load(os.path.join(DATA_DIR, 'train_data.npz'))
    val_npz = np.load(os.path.join(DATA_DIR, 'val_data.npz'))
    test_npz = np.load(os.path.join(DATA_DIR, 'test_data.npz'))

    X_train = torch.tensor(train_npz['X'], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_npz['y'], dtype=torch.long)
    rec_train = train_npz['record_ids']

    X_val = torch.tensor(val_npz['X'], dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(val_npz['y'], dtype=torch.long)
    rec_val = val_npz['record_ids']

    X_test = torch.tensor(test_npz['X'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_npz['y'], dtype=torch.long)
    rec_test = test_npz['record_ids']

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Compute class weights
    class_counts = torch.bincount(y_train, minlength=4)
    total_samples = len(y_train)
    weights = total_samples / (4.0 * class_counts.float() + 1e-5)
    weights = weights / weights.min()  # Normalize so min is 1.0

    print(f"Data Loaded:")
    print(f"  Train: {len(X_train):,} beats, Counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}")
    print(f"  Val:   {len(X_val):,} beats, Counts: {dict(zip(CLASS_NAMES, torch.bincount(y_val, minlength=4).tolist()))}")
    print(f"  Test:  {len(X_test):,} beats, Counts: {dict(zip(CLASS_NAMES, torch.bincount(y_test, minlength=4).tolist()))}")
    print(f"  Class Weights (Normalized): {[round(w, 2) for w in weights.tolist()]}")

    return train_loader, val_loader, test_loader, weights, rec_test, y_test

def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            if criterion:
                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(all_labels) if criterion else 0.0
    return avg_loss, np.array(all_preds), np.array(all_labels)

def train_single_model(seed, weighted=True, max_epochs=30, patience=5, batch_size=256):
    set_seed(seed)
    train_loader, val_loader, test_loader, class_weights, rec_test, y_test = load_data(batch_size=batch_size)

    model = ECGModel(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if weighted:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        save_dir = os.path.join(MODELS_DIR, 'weighted')
        tag = f"weighted_seed_{seed}"
    else:
        criterion = nn.CrossEntropyLoss()
        save_dir = os.path.join(MODELS_DIR, 'unweighted')
        tag = f"unweighted_seed_{seed}"

    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"model_{tag}.pth")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Training Model: {tag} (Max Epochs: {max_epochs}, Patience: {patience})")
    print(f"{'='*60}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_samples += len(y_batch)

        train_loss /= train_samples
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion=criterion)
        val_report = classification_report(val_labels, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        val_macro_f1 = val_report['macro avg']['f1-score']

        dur = time.time() - t0
        print(f"Epoch {epoch:02d}/{max_epochs:02d} [{dur:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}", flush=True)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_macro_f1': val_macro_f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model checkpoint (Val Loss: {best_val_loss:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (Best Val Loss: {best_val_loss:.4f})", flush=True)
                break

    # Load best model for test evaluation
    print(f"\nEvaluating best model on DS2-Test ({len(y_test):,} beats)...", flush=True)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion=criterion)

    report_text = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    report_dict = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n--- DS2-TEST RESULTS ({tag}) ---")
    print(report_text, flush=True)
    print("Confusion Matrix (True \ Pred):")
    print(cm, flush=True)

    result_summary = {
        'tag': tag,
        'seed': seed,
        'weighted': weighted,
        'best_val_loss': best_val_loss,
        'stopped_epoch': len(history),
        'test_metrics': report_dict,
        'confusion_matrix': cm.tolist(),
        'history': history
    }

    out_json = os.path.join(METRICS_DIR, f"results_{tag}.json")
    with open(out_json, 'w') as f:
        json.dump(result_summary, f, indent=2)

    return result_summary

if __name__ == '__main__':
    # Train the first model (Seed 42, Weighted) to inspect initial results
    print("Executing First Model Training (Seed 42, Class-Weighted)...", flush=True)
    train_single_model(seed=42, weighted=True, max_epochs=25, patience=5)
