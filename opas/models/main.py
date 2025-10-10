import math
import torch
import torch.nn as nn


class TransformInput(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform(x)


class LamModel(nn.Module):
    def __init__(self, M, N, stagger):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(stagger+1, 4, kernel_size=(1, N//4)),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=(1, N//8)),
            nn.MaxPool2d((1,2)),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=(1, N//16)),
            nn.MaxPool2d((1,2)),
            nn.ReLU(),
            nn.Conv2d(4, M, kernel_size=(1, N//16)),
            nn.AdaptiveAvgPool2d((1,20)),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear(20, M),
            nn.ReLU(),
            nn.Linear(M, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class ScoreModel(nn.Module) :
    def __init__(self, indim=2, outdim=1) :
        super().__init__()
        self.weight = nn.Parameter(torch.randn(indim, outdim))
    def forward(self, x) :
        logits = torch.matmul(x, torch.exp(self.weight))
        return nn.Sigmoid()(logits)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DummyScoreModel(nn.Module):
    def __init__(self, indim=2, outdim=1) :
        super().__init__()
        self.weight = nn.Parameter(torch.randn(indim, outdim))
    def forward(self, x) :
        logits = x.sum(dim=-1)
        return nn.Sigmoid()(logits)


class LamModel4LongSeq(nn.Module):
    def __init__(self, M, N, stagger):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(stagger+1, 4, kernel_size=(1, N//4)),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=(1, N//16)),
            nn.MaxPool2d((1,2)),
            nn.ReLU(),
            # nn.Conv2d(4, 4, kernel_size=(1, N//16)),
            # nn.MaxPool2d((1,2)),
            # nn.ReLU(),
            nn.Conv2d(4, M, kernel_size=(1, N//16)),
            nn.AdaptiveAvgPool2d((1,20)),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear(20, M),
            nn.ReLU(),
            nn.Linear(M, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class Attention_Layer(nn.Module):
    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.w = nn.Linear(
            in_features=n_feats,
            out_features=n_feats
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        w = self.w(X)
        output = F.softmax(torch.mul(X, w), dim=1)
        return output