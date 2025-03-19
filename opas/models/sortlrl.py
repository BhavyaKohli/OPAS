import torch
import torch.nn as nn


class SortLRL(nn.Module):
    def __init__(self, indim, seq_len, latent, outdim):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, indim))
        self.lrl = nn.Sequential(
            nn.Linear(seq_len, latent),
            nn.ReLU(),
            nn.Linear(latent, outdim)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, indim)
        # output: (batch_size, outdim)
        proj = (x @ self.alpha.T).squeeze(-1)
        proj = torch.sort(proj, dim=-1)
        return self.lrl(proj.values)