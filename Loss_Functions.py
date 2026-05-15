import sys
import os

import numpy as np

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

# from geomloss import SamplesLoss

import matplotlib.pyplot as plt

###############################################################################

def MSELoss_(x, x_, z_):
    # x:  (B, D...)
    # x_: (B, D...)
    # z_: (B, d...)
    assert x.shape == x_.shape, f"{x.shape}, {x_.shape}"
    
    L = ((x_ - x)**2).mean()     # (1,)
    return L
    
    
def MAELoss_(x, x_, z_):
    # x:  (B, D...)
    # x_: (B, D...)
    # z_: (B, d...)
    assert x.shape == x_.shape, f"{x.shape}, {x_.shape}"
    
    L = ((x_ - x).abs()).mean()     # (1,)
    return L
    
###############################################################################
# Regularization

# def VAE_Loss_(x, x_, z_):
#     """
#     Variational Autoencoder loss combining reconstruction loss and KL divergence.
#     Assumes z_ contains concatenated [μ_, logσ2_] along the last dimension.

#     x_: reconstruction
#     x: original input
#     z_: tensor of shape (..., 2D), with last dim containing [mu, logvar]
#     """
#     # x:  (B, D...)
#     # x_: (B, D...)
#     # z_: (B, 2C...)
    
    
#     # self.recon_loss = nn.MSELoss()

    
#     # Reconstruction loss
#     L2 = self.recon_loss(x_, x)                     # (1,)

#     # Split latent into mu and logvar
#     μ_, logσ2_ = tc.chunk(z_, 2, dim=-1)            # 2 (B, C...)

#     # KL divergence between N(mu, sigma^2) and N(0, 1)
#     KLD = -0.5 * tc.sum(1 + logσ2_ - μ_.pow(2) - logσ2_.exp(), dim=-1)      # (B,)
#     KLD = KLD.mean()                                                        # (1,)
    
#     L = L2 + self.beta * KLD        # (1,)

    # Compute VAE loss: reconstruction + beta * KL divergence.
    
    # Args:
    #     x_recon: reconstructed input
    #     x: original input
    #     z: latent vector with concatenated [mu, logvar]
    #     beta: weight for KL divergence term
    # """
    # # Reconstruction loss (MSE)
    # recon_loss = F.mse_loss(x_recon, x)

    # # Split z into mean and log variance
    # mu, logvar = torch.chunk(z, 2, dim=-1)

    # # KL divergence between N(mu, sigma^2) and N(0,1)
    # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    # kld = kld.mean()

    # return recon_loss + beta * kld

    # return L
    
    

def Gaussian_Regularization_(z_):
    # z_: (B, d...)
    B, d = z_.shape[0], z_.shape[1:]
    
    μ = z_.mean(dim=0)                  # (d...)
    
    if B <= 1:
        σ = tc.ones(d)                     # (d...)
    else:
        σ = z_.std(dim=0)               # (d...)
    
    L = (μ**2 + (σ - 1)**2).mean()      # (1,)
    
    return L


###############################################################################
# Point Cloud Losses

def Chamfer_Distance_(X1, X2):
    """
    Computes the Chamfer Distance between two point clouds.
    
    Args:
    - X1: Tensor of shape (B, N, D)  # First point cloud
    - X2: Tensor of shape (B, M, D)  # Second point cloud

    Returns:
    - C: The Chamfer Distance between the two point clouds.
    """
    
    D = ((X1[:, :, None] - X2[:, None]) ** 2).sum(dim=-1)    # (B, N, M)
    
    # Find min distance from each point in pc1 to pc2 and vice versa
    m1 = D.min(dim=2)[0]                        # (B, N)
    m2 = D.min(dim=1)[0]                        # (B, M)
    
    C = m1.mean(axis=1) + m2.mean(axis=1)       # (B,)
    C = C.mean()                                # (1,)
    
    return C


def Sinkhorn_Loss_(X1, X2):
    """
    Computes the Sinkhorn Loss between two point clouds.
    
    Args:
    - X1: Tensor of shape (B, N, D)  # First point cloud
    - X2: Tensor of shape (B, M, D)  # Second point cloud

    Returns:
    - L: The Sinkhorn Distance between the two point clouds.
    """
    Loss_ = SamplesLoss("sinkhorn", p=2, blur=0.05)  # blur ≈ ε
    L = Loss_(X1, X2).mean()                   # (1,)
    return L

###############################################################################

