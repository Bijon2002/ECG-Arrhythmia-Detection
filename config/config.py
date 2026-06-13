import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_CONFIG = {
    'num_classes': 5,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50
}

DATA_CONFIG = {
    'database_name': 'mitdb'
}

