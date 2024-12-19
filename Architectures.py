import sys
import os

import numpy as np


import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

###############################################################################

class Residual_Block_(nn.Module):
    def __init__(self, size:int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: tc.tensor):
        return x + self.act(self.ff(x))


def Sinusoidal_Embedding_(x:tc.tensor, E:int=256):
    # x (B, 1)
    B = x.shape[0]
    
    d_ = tc.arange(E, dtype=tc.float32)                # (E,)
    k = tc.exp(-d_/E * tc.log(tc.tensor(10000.0)))     # (E,)
    kx = k[None, :] * x                                # (B, E)
    p = tc.zeros((B, E), dtype=tc.float32)             # (B, E)
    p[:, 0::2] = tc.sin(kx[:, 0::2])                   # (B, E)
    p[:, 1::2] = tc.cos(kx[:, 1::2])                   # (B, E)
    return p


class Model_ERNN__(nn.Module):
    
    def __init__(self, D_in:int, D_out:int, E:int):
        super().__init__()

        self.D_in = D_in
        self.D_out = D_out     
        self.E = E
        
        hidden_size = 128
        hidden_layers = 10
        layers = [nn.Linear(D_in, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Residual_Block_(hidden_size))
        layers.append(nn.Linear(hidden_size, D_out))
        self.layers = nn.Sequential(*layers)

        return

    def forward(self, x, t):
        
        p_t = Sinusoidal_Embedding_(t, E=self.E)                  # (B, E)
        p_x = Sinusoidal_Embedding_(x[:, 0:1], E=self.E)          # (B, E) 
        p_y = Sinusoidal_Embedding_(x[:, 1:], E=self.E)           # (B, E) 
        
        x_p = tc.cat([p_t, p_x, p_y], axis=-1)                    # (B, D_in)  

        y = self.layers(x_p)                                      # (B, D_out)          
        return y
        

class Image_AE_(nn.Module):
    def __init__(self, H=128, W=128, C=3, L=256):
        super(Image_AE_, self).__init__()
        # X: (B, C, H, W)
        self.H = H
        self.W = W
        self.C = C
        self.L = L
        
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=4, stride=2, padding=1),               # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),            # (B, 256, H/8, W/8)
            nn.ReLU(),
            nn.Flatten(),                                                       # (B, 256 * H/8 * W/8)
            nn.Linear(256 * H//8 * W//8, L),                                    # (B, L)
        )
        
        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(L, 256 * H//8 * W//8),                                    # (B, 256 * H/8 * W/8)
            nn.Unflatten(1, (L, H//8, W//8)),                                   # (B, 256, H/8, W/8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, C, kernel_size=4, stride=2, padding=1),      # (B, C, H, W)
            nn.Sigmoid(),  # Output pixel values in [0, 1]
        )
        
    def forward(self, x):
        # (B, C, H, W)
        
        z_ = self.Encoder(x)     # (B, L)
        x_ = self.Decoder(z_)     # (B, C, H, W)
        
        return x_, z_

# Example usage
if __name__ == "__main__":
    model = Image_AE_()
    sample_input = tc.randn(8, 3, 128, 128)  # Batch of 8 images
    reconstructed = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")







