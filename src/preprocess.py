import librosa
import soundfile as sf
import os
from pathlib import Path

def preprocess_audio(input_dir, output_dir, target_sr=16000):
    if not os.path.exists(input_dir):
        print(f"❌ Error: {input_dir} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    files = list(Path(input_dir).rglob("*.wav")) + list(Path(input_dir).rglob("*.flac"))
    
    print(f"🧹 Processing {len(files)} files from {input_dir}...")
    
    for i, file_path in enumerate(files):
        try:
            # 1. Load and Resample (Handles both .wav and .flac)
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # 2. VAD: Trim silence
            yt, _ = librosa.effects.trim(y, top_db=20)
            
            # 3. Normalize
            yt = librosa.util.normalize(yt)
            
            # 4. Save
            out_name = f"sample_{i}.wav"
            sf.write(os.path.join(output_dir, out_name), yt, target_sr)
            
            if i % 100 == 0:
                print(f"✅ Processed {i} files...")
        except Exception as e:
            print(f"⚠️ Error on {file_path}: {e}")

if __name__ == "__main__":
    # Point these to your local dataset folders
    preprocess_audio("dataset/real", "data/processed/real")
    preprocess_audio("dataset/fake", "data/processed/fake")