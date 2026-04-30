import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import DeepScanModel # Your CNN-BiLSTM architecture
import numpy as np
import os

def train_model():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Initializing Training Pipeline on {device}...")
    
    # 1. Load Augmented Features
    try:
        features = np.load("features_seq.npy")
        labels = np.load("labels.npy")
        print(f"📦 Loaded {len(features)} signatures.")
    except Exception as e:
        print("❌ Could not load features. Run python extract_features.py first!")
        return

    # 2. Validation Split (80/20)
    np.random.seed(42) # For reproducibility
    indices = np.random.permutation(len(features))
    split_idx = int(0.8 * len(features))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train = torch.from_numpy(features[train_idx]).float()
    y_train = torch.from_numpy(labels[train_idx]).float().unsqueeze(1)
    
    # If the dataset is too small, use training data for validation so it doesn't crash
    if len(val_idx) > 0:
        X_val = torch.from_numpy(features[val_idx]).float()
        y_val = torch.from_numpy(labels[val_idx]).float().unsqueeze(1)
    else:
        print("⚠️ Dataset extremely small. Using training data for validation.")
        X_val, y_val = X_train, y_train
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

    # 3. Setup Model, Loss, Optimizer, and Scheduler
    model = DeepScanModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # 4. Training Loop with Early Stopping
    epochs = 50
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    print("⏳ Starting Training...")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate Accuracy
                predictions = (outputs > 0.5).float()
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}")
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "models/deep_scan_v1.pth")
            print("   🌟 New Best Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered. No improvement in {patience} epochs.")
                break

    print("✨ Training Complete. Best model is saved at models/deep_scan_v1.pth")

if __name__ == "__main__":
    train_model()