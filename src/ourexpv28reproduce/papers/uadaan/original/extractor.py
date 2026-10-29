import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(640),
            nn.Linear(640,256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.Dropout(p=0.2)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x