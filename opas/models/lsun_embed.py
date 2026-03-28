# -*- coding: utf-8 -*-

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 256, 256]
        # Output size: [batch, 3, 256, 256]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 128, 128]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 64, 64]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 32, 32]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 16, 16], 
            nn.ReLU(),
            nn.Conv2d(96, 96, 4, stride=2, padding=1),           # [batch, 96, 8, 8]
            nn.ReLU(),
            nn.Conv2d(96, 96, 4, stride=2, padding=1),           # [batch, 96, 4, 4], embed size = 96*16 = 1536 dim
            nn.ReLU(),
            nn.Conv2d(96, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2], embed size = 96*4 = 384 dim
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 96, 4, stride=2, padding=1),  # [batch, 96, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 4, stride=2, padding=1),  # [batch, 96, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 4, stride=2, padding=1),  # [batch, 96, 16, 16]
            nn.ReLU(),
			nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 32, 32]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 64, 64]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 256, 256]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
