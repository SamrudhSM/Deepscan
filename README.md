# 🕵️ DeepScan Forensics: AI Voice Detection Platform

**DeepScan Forensics** is a full-stack, production-ready AI application designed to detect synthetic, AI-generated voices (deepfakes) with incredible accuracy. It utilizes advanced machine learning models (HuBERT + Dense-BiLSTM) to analyze acoustic consistency and temporal artifacts, cross-referencing results against a high-dimensional FAISS vector database.

Built from the ground up for a premium user experience, the system features a sleek React (Vite) frontend with real-time WebSocket communication to a lightning-fast FastAPI Python backend.

---

## 🌟 Key Features

- **Dual-Engine Analysis**: Combines **HuBERT** feature extraction for microscopic phonetic anomalies with a **Dense-BiLSTM** neural network to detect unnatural prosody over time.
- **FAISS Forensic Memory**: Queries audio signatures against a high-dimensional vector database for nearest-match verification.
- **Live Stream Detection**: Supports real-time "Live Mic" recording using WebSockets, processing audio chunks on the fly.
- **Data Augmentation Engine**: A highly optimized pipeline that multiplies the dataset size using synthetic noise, pitch shifting, and speed perturbation to drastically improve model generalization.
- **Premium Frontend UX**: A beautiful, dark-mode glassmorphism interface built with React, Vite, and modern CSS.

---

## 🏗️ Architecture

- **Frontend**: React, Vite, TypeScript, Vanilla CSS
- **Backend**: FastAPI, WebSockets, Uvicorn
- **AI Models**: PyTorch, Transformers (Hugging Face HuBERT), Librosa
- **Database**: FAISS (Facebook AI Similarity Search)

---

## 🚀 Getting Started

### 1. Backend (FastAPI & PyTorch)

Navigate to the root directory and activate your virtual environment:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if you haven't already)
pip install fastapi uvicorn torch transformers librosa faiss-cpu

# Start the FastAPI server
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at `http://127.0.0.1:8000`. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 2. Frontend (React / Vite)

In a new terminal window, start the frontend application:

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

The application will be running at `http://localhost:5173`.

---

## 🧠 Upgrading the AI Brain (Training)

The neural network is designed to be trained efficiently, even on Apple Silicon (M1/M2) using the `mps` backend.

1. **Add Data**: Place raw `.wav` or `.flac` audio files into `data/processed/real` and `data/processed/fake`.
2. **Extract & Augment**: Run the feature extractor. This script will automatically augment your audio (noise, pitch, speed) to create a robust dataset.
   ```bash
   cd src
   python extract_features.py
   ```
3. **Train the Model**: Run the training script. It features an 80/20 validation split, early stopping, and a learning rate scheduler to prevent overfitting.
   ```bash
   python train.py
   ```

---

## 📄 License

This project is open-source and available for educational and portfolio purposes.
