import torch
import torch.nn as nn

class DeepScanModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        # HuBERT DNA is now a temporal sequence: (batch, 50, 768)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # BiLSTM to process the features over time
        self.lstm = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, 50, 768)
        x = self.feature_extractor(x) # outputs (batch, 50, 512)
        
        # LSTM processes the 50 steps
        _, (hn, _) = self.lstm(x)
        
        # Combine both directions of the LSTM's final hidden state
        out = torch.cat((hn[0], hn[1]), dim=1)
        return self.classifier(out)