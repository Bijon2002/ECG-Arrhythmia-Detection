print('Testing ECG setup...')
print()
# Test 1: Import utils
from src.utils import get_device
print('? Utils import works')
# Test 2: Import config
from config.config import MODEL_CONFIG
print('? Config import works')
# Test 3: Check device
device = get_device()
print('? Device:', device)
# Test 4: Check config values
print('? Classes:', MODEL_CONFIG['num_classes'])
print()
print('?? All tests passed! Setup is working!')
