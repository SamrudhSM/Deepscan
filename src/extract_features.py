import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import librosa
import numpy as np
import os
from pathlib import Path

class DNAExtractor:
    def __init__(self):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"🧠 Loading HuBERT on {self.device}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
        self.model.eval()

    def get_dna_from_array(self, y, sr=16000):
        inputs = self.processor(y, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            
        seq = outputs.last_hidden_state # shape: (1, seq_len, 768)
        
        # 1. Flat DNA for FAISS (mean pooling)
        dna_flat = seq.mean(dim=1).cpu().numpy() # shape: (1, 768)
        
        # 2. Sequential DNA for BiLSTM (Adaptive Pooling to 50 steps)
        seq = seq.transpose(1, 2).cpu() # Move to CPU to avoid MPS Adaptive Pool bug
        import torch.nn.functional as F
        seq_pooled = F.adaptive_avg_pool1d(seq, 50) # shape: (1, 768, 50)
        seq_pooled = seq_pooled.transpose(1, 2) # shape: (1, 50, 768)
        dna_seq = seq_pooled.numpy()
        
        return dna_flat, dna_seq

    def get_dna(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)
        return self.get_dna_from_array(y, sr)

    def get_dna_from_bytes(self, audio_bytes):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            return self.get_dna(tmp_path)
        except Exception as e:
            print(f"Error extracting DNA from bytes: {e}")
            raise e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

def augment_audio(y, sr=16000):
    """Generate augmented versions of the audio array to multiply dataset size."""
    versions = [y] # Original
    
    # 1. Add background noise
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amp * np.random.normal(size=y.shape[0])
    versions.append(y_noise)
    
    # 2. Pitch shift (slightly lower)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5)
    versions.append(y_pitch)
    
    # 3. Time stretch (slightly faster)
    y_speed = librosa.effects.time_stretch(y, rate=1.1)
    versions.append(y_speed)
    
    return versions

def process_all_dna():
    extractor = DNAExtractor()
    features_flat_list = []
    features_seq_list = []
    labels = [] # 0 for Real, 1 for Fake

    def process_directory(directory, label_val, label_name):
        files = list(Path(directory).glob("*.wav")) + list(Path(directory).glob("*.flac"))
        if not files:
            print(f"⚠️ No files found in {directory}")
            return
            
        print(f"🧬 Extracting Dual-DNA from {len(files)} {label_name} samples (with 4x augmentation)...")
        for f in files:
            try:
                y, sr = librosa.load(f, sr=16000)
                augmented_versions = augment_audio(y, sr)
                
                for aug_y in augmented_versions:
                    dna_flat, dna_seq = extractor.get_dna_from_array(aug_y, sr)
                    features_flat_list.append(dna_flat)
                    features_seq_list.append(dna_seq)
                    labels.append(label_val)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    # Process Real
    process_directory("data/processed/real", 0, "Real")
    
    # Process Fake
    process_directory("data/processed/fake", 1, "Fake")

    if not features_flat_list:
        print("❌ No features extracted. Please ensure audio files exist in data/processed/real and data/processed/fake")
        return

    # Save everything for Training and FAISS
    np.save("features_flat.npy", np.vstack(features_flat_list))
    np.save("features_seq.npy", np.vstack(features_seq_list))
    np.save("labels.npy", np.array(labels))
    print(f"✅ Data Augmentation & Extraction Complete. Generated {len(labels)} total signatures!")

if __name__ == "__main__":
    process_all_dna()