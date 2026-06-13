import os
import wfdb
import sys

# Add project root to path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import RAW_DATA_DIR, DATA_CONFIG

def main():
    database = DATA_CONFIG['database_name']
    dl_dir = RAW_DATA_DIR
    
    # Ensure directory exists
    os.makedirs(dl_dir, exist_ok=True)
    
    print(f"Downloading PhysioNet database '{database}' to '{dl_dir}'...")
    print("This might take a few minutes (approx 72 MB)...")
    
    try:
        # Download the entire database
        wfdb.dl_database(database, dl_dir=dl_dir)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error occurred during download: {e}")

if __name__ == "__main__":
    main()
