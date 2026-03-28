import torch
import torch.nn as nn

class DeepSetModel(nn.Module):
    def __init__(self, indim, latent, outdim, aggr="sum"):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(indim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent)
        )
        
        self.alpha = nn.Parameter(torch.randn(1, latent), requires_grad=True)

        self.phi = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, outdim)
        )

        self.aggr = aggr

    def forward(self, x):
        # expect batch x M/N x indim
        # output batch x outdim
        
        x = self.rho(x)   # batch x M/N x latent
        aggr = getattr(self, "aggr", "sum")
        if aggr == "sum":
            x = torch.sum(x, dim=1) # batch x latent
        elif aggr == "max":
            x = torch.max(x, dim=1).values
        x = x + self.alpha
        x = self.phi(x)  # batch x outdim
        return x
    

class DeepSetwAttnMask(nn.Module):
    def __init__(self, indim, latent, outdim):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(indim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, latent)
        )
        
        self.alpha = nn.Parameter(torch.randn(1, latent), requires_grad=True)

        self.phi = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, outdim)
        )

    def forward(self, x, attn_mask=None):
        # expect batch x M/N x indim
        # output batch x outdim

        if attn_mask is None:
            attn_mask = torch.ones(x.size(0), x.size(1), device=x.device)
        
        x = self.rho(x)   # batch x M/N x latent
        x = x * attn_mask.unsqueeze(-1)  # apply mask to remove contribution from padded elements
        x = x.sum(dim=1)  # batch x latent

        x = x + self.alpha
        x = self.phi(x)  # batch x outdim
        return x
    

class DeepSetLamModel(nn.Module):
    def __init__(self, indim, latent, outdim, m, share_deepset=False):
        super().__init__()
        self.deepsetq = DeepSetwAttnMask(indim, latent, outdim)

        if share_deepset:
            self.deepsetc = self.deepsetq
        else:
            self.deepsetc = DeepSetwAttnMask(indim, latent, outdim)

        self.attn = nn.MultiheadAttention(embed_dim=outdim, num_heads=4, batch_first=True, dropout=0.1)
        self.output = nn.Linear(outdim, m)

    def forward(self, q, c, m_a):
        """
        q: batch x M x indim
        c: batch x N x indim

        both q and c become batch x outdim after deepsets
        
        for q||c ==> batch x outdim x 2 ---OR--- batch x (2*outdim)
        """
        q = self.deepsetq(q)        # batch x outdim
        c = self.deepsetc(c, m_a)   # batch x outdim

        attn_output, _ = self.attn(q, c, c)
        lambdas = self.output(attn_output)  # batch x m
        lambdas = torch.abs(lambdas)    # need to be positive
        return lambdas
