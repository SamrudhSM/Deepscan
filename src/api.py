import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
import torch
import numpy as np
import sys
import faiss

# Ensure local imports work correctly when running from root
sys.path.append(os.path.join(os.path.dirname(__file__)))

from extract_features import DNAExtractor
from model import DeepScanModel

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DeepScan Audio Forensics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🧠 Initializing Extractor and Model for API...")
extractor = DNAExtractor()
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = DeepScanModel().to(device)

# Provide the correct absolute path fallback or relative to root
model_path = os.path.join(os.path.dirname(__file__), "../models/deep_scan_v1.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

print("🔍 Loading FAISS Forensic Memory...")
index_path = os.path.join(os.path.dirname(__file__), "../forensic_voices.index")
labels_path = os.path.join(os.path.dirname(__file__), "../labels.npy")

try:
    faiss_index = faiss.read_index(index_path)
    faiss_labels = np.load(labels_path)
    print(f"✅ FAISS Memory Loaded: {faiss_index.ntotal} signatures.")
except Exception as e:
    print(f"⚠️ FAISS Memory could not be loaded: {e}")
    faiss_index = None
    faiss_labels = None

print("✅ API Ready")

@app.post("/detect")
async def detect_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily to process through librosa
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Extract DNA
        dna_flat, dna_seq = extractor.get_dna(temp_path)
        
        # 2. Neural Inference
        with torch.no_grad():
            dna_tensor = torch.from_numpy(dna_seq).float().to(device)
            if len(dna_tensor.shape) == 2:
                dna_tensor = dna_tensor.unsqueeze(0) # (1, 50, 768)
            prediction = model(dna_tensor).item()
            
        # 3. FAISS Memory Search
        faiss_result = None
        if faiss_index is not None:
            # Search for the 1 nearest neighbor
            D, I = faiss_index.search(dna_flat.astype('float32'), 1)
            nearest_idx = I[0][0]
            distance = float(D[0][0])
            nearest_label = int(faiss_labels[nearest_idx])
            
            label_str = "DEEPFAKE" if nearest_label == 1 else "HUMAN"
            
            # Simple heuristic for similarity percentage based on L2 distance
            similarity_score = np.exp(-distance / 150) # Tweak divisor for more reasonable %
            similarity_percent = float(similarity_score * 100)
            
            faiss_result = {
                "match_verdict": label_str,
                "distance": distance,
                "similarity": min(99.9, max(0.1, similarity_percent))
            }
            
        # 4. Format Response
        label = "DEEPFAKE" if prediction > 0.5 else "HUMAN"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        return {
            "filename": file.filename,
            "verdict": label,
            "confidence": confidence,
            "faiss_memory": faiss_result
        }
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket Client Connected")
    session_audio = bytearray()
    try:
        while True:
            data = await websocket.receive_bytes()
            session_audio.extend(data)
            
            # Require at least 1.5 seconds of audio (3 chunks of 500ms) before predicting
            # to give the model enough context to find artifacts, rather than predicting HUMAN on short noise.
            if len(session_audio) < 15000: # rough byte estimate for 1.5s of webm opus
                continue
                
            try:
                # 1. Extract DNA from accumulated streaming audio
                dna_flat, dna_seq = extractor.get_dna_from_bytes(bytes(session_audio))
                
                # 2. Neural Inference
                with torch.no_grad():
                    dna_tensor = torch.from_numpy(dna_seq).float().to(device)
                    if len(dna_tensor.shape) == 2:
                        dna_tensor = dna_tensor.unsqueeze(0)
                    prediction = model(dna_tensor).item()
                
                # 3. FAISS Memory Search
                faiss_result = None
                if faiss_index is not None:
                    D, I = faiss_index.search(dna_flat.astype('float32'), 1)
                    nearest_idx = I[0][0]
                    distance = float(D[0][0])
                    nearest_label = int(faiss_labels[nearest_idx])
                    
                    label_str = "DEEPFAKE" if nearest_label == 1 else "HUMAN"
                    similarity_score = np.exp(-distance / 150)
                    similarity_percent = float(similarity_score * 100)
                    
                    faiss_result = {
                        "match_verdict": label_str,
                        "distance": distance,
                        "similarity": min(99.9, max(0.1, similarity_percent))
                    }
                
                label = "DEEPFAKE" if prediction > 0.5 else "HUMAN"
                confidence = prediction if prediction > 0.5 else (1 - prediction)
                
                await websocket.send_json({
                    "verdict": label,
                    "confidence": confidence,
                    "faiss_memory": faiss_result
                })
            except Exception as e:
                print(f"⚠️ Error processing audio chunk: {e}")
                await websocket.send_json({"error": "Failed to process audio chunk"})
    except WebSocketDisconnect:
        print("🔌 WebSocket Client Disconnected")
