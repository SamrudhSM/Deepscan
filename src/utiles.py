import os
import glob

def check_dataset_balance(base_path):
    # This will help you verify you have 1000 real and 1000 fake
    real_count = len(glob.glob(f"{base_path}/real/**/*.*", recursive=True))
    fake_count = len(glob.glob(f"{base_path}/fake/**/*.*", recursive=True))
    
    print(f"📊 Dataset Status:")
    print(f"   - Real Samples: {real_count}")
    print(f"   - Fake Samples: {fake_count}")
    return real_count == fake_count

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)