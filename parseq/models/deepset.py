import torch
import torch.nn as nn

class DeepSetModel(nn.Module):
    def __init__(self, indim, latent, outdim):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(indim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent)
        )
        
        self.alpha = nn.Parameter(torch.randn(1, latent), requires_grad=True)

        self.phi = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, outdim)
        )

    def forward(self, x):
        # expect batch x M/N x indim
        # output batch x outdim
        
        x = self.rho(x)   # batch x M/N x latent
        x = torch.sum(x, dim=1) # batch x latent
        x = x + self.alpha
        x = self.phi(x)  # batch x outdim
        return x