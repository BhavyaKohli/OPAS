import torch
import torch.nn as nn
from audio_encoders_pytorch import Encoder1d, MelE1d


class Conv1dTS(nn.Module):
    def __init__(self, n_ch, latent):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_ch, kernel_size=200),
            nn.BatchNorm1d(n_ch),
            nn.MaxPool1d(2),
            nn.Conv1d(n_ch, 2*n_ch, kernel_size=16),
            nn.BatchNorm1d(2*n_ch),
            nn.MaxPool1d(2),
            nn.Conv1d(2*n_ch, 4*n_ch, kernel_size=16),
            nn.BatchNorm1d(4*n_ch),
            nn.AdaptiveAvgPool1d(1024)
        )

        self.lin = nn.Sequential(
            nn.Linear(1024*4*n_ch, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, latent)
        )

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, 1, orig_shape[-1])
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        x = x.reshape(*orig_shape[:-1], x.shape[-1])
        return x


class EncConv1dTS(nn.Module):
    def __init__(self, latent, samp_rate=4000):
        super().__init__()
        self.model = Encoder1d(
            in_channels=1, 
            channels=32, 
            multipliers=[1, 1, 2, 2], 
            factors=[4, 4, 4], 
            num_blocks=[2, 2, 2]
        )
        
        dummy_ip = torch.randn(7, 1, samp_rate)
        sz = self.model(dummy_ip).flatten(start_dim=1).shape[-1]

        self.lin = nn.Sequential(
            nn.Linear(sz, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent)
        )
    
    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, 1, orig_shape[-1])
        x = self.model(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        x = x.reshape(*orig_shape[:-1], x.shape[-1])
        return x


class MelConv1dTS(nn.Module):
    def __init__(self, latent, samp_rate=4000):
        super().__init__()
        self.model = MelE1d(
            in_channels=1, 
            mel_channels=32,
            channels=32, 
            multipliers=[1, 1, 2, 2], 
            factors=[4, 4, 4], 
            num_blocks=[2, 2, 2]
        )
        
        dummy_ip = torch.randn(7, 1, samp_rate)
        sz = self.model(dummy_ip).flatten(start_dim=1).shape[-1]

        self.lin = nn.Sequential(
            nn.Linear(sz, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent)
        )
    
    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, 1, orig_shape[-1])
        x = self.model(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        x = x.reshape(*orig_shape[:-1], x.shape[-1])
        return x