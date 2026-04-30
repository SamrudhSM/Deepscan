import torch
import numpy as np
from extract_features import DNAExtractor
from model import DeepScanModel

def detect(audio_path):
    # 1. Extract DNA
    extractor = DNAExtractor()
    dna_flat, dna_seq = extractor.get_dna(audio_path)
    
    # 2. Ask the CNN-BiLSTM
    model = DeepScanModel()
    model.load_state_dict(torch.load("models/deep_scan_v1.pth"))
    model.eval()
    
    with torch.no_grad():
        dna_tensor = torch.from_numpy(dna_seq).float().unsqueeze(0)
        prediction = model(dna_tensor).item()
    
    # 3. Output
    label = "🔴 DEEPFAKE" if prediction > 0.5 else "🟢 HUMAN"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    print(f"\n--- Forensic Result ---")
    print(f"File: {audio_path}")
    print(f"Verdict: {label}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    # Test it on one of your processed files!
    detect("data/processed/real/sample_0.wav")