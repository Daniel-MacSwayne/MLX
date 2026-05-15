import sys
import os

import numpy as np
import math
import scipy.fftpack as fft

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, dense_diff_pool
# from torch_geometric.nn import DiffPool
from torch_geometric.data import Data, Batch

import matplotlib.pyplot as plt

from Graphs import *

###############################################################################
# Activation Functions

Activation_Functions = {'ReLU':     nn.ReLU(),
                        'Sigmoid':  nn.Sigmoid(),
                        'Tanh':     nn.Tanh(),
                        'Softmax':  nn.Softmax(),
                        'GELU':     nn.GELU(),
                        'SiLU':     nn.SiLU(),
                        }

class Softmax_Temperature_(nn.Module):
    def __init__(self, t=1.0):
        super(Softmax_Temperature_, self).__init__()
        self.t = t

    def forward(self, x, dim=-1):
        # Apply temperature scaling to the logits
        return F.softmax(x / self.t, dim=dim)

###############################################################################
# Embeddings

def Grid_Position_Embedding_(H, W):
    """
    Normalized Cartesian Coordinates.
    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        D (int): Embedding dimension (must be even).
    Returns:
        Tensor of shape (2, H, W).
    """
    # Normalized coordinates
    x_ = tc.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)     # (H, W)
    y_ = tc.linspace(0, 1, H).unsqueeze(1).repeat(1, W)     # (H, W)
    Grid = tc.stack((x_, y_), dim=0)                        # (2, H, W)

    return Grid


def Sinusoidal_Embedding_1D_(x:tc.tensor, E:int=256):
    """
    Generate sinusoidal positional encodings for a 1D.

    Args:
        E (int): Total embedding dimension (must be even).

    Returns:
        torch.Tensor: Positional encoding tensor of shape (B, E).
    """
    # x (B,)
    B = x.shape[0]
    
    d_ = tc.arange(E, dtype=tc.float32)                # (E,)
    k = tc.exp(-d_/E * tc.log(tc.tensor(10000.0)))     # (E,)
    kx = k[None, :] * x[:, None]                       # (B, E)
    p = tc.zeros((B, E), dtype=tc.float32)             # (B, E)
    p[:, 0::2] = tc.sin(kx[:, 0::2])                   # (B, E)
    p[:, 1::2] = tc.cos(kx[:, 1::2])                   # (B, E)
    return p


def Sinusoidal_Embedding_2D_(H, W, E):
    """
    Generate sinusoidal positional encodings for a 2D image.

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        D (int): Total embedding dimension (must be even).

    Returns:
        torch.Tensor: Positional encoding tensor of shape (H, W, D).
    """
    assert E % 2 == 0, "Embedding dimension D must be even."

    # Create grid of coordinates
    x_ = tc.arange(W)[None].expand(H, -1)               # (H, W)
    y_ = tc.arange(H)[:, None].expand(-1, W)            # (H, W)
    
    # Compute embedding dimensions
    E_half = E // 2
    div_term = tc.exp(tc.arange(0, E_half, 2) * -(math.log(10000.0) / E_half))  # (E/2,)
    
    # Apply sin/cos for x-coordinates
    x_ = x_[..., None] * div_term                       # (H, W, E/2)
    p_x = tc.cat([x_.sin(), x_.cos()], dim=-1)          # (H, W, E)

    # Apply sin/cos for y-coordinates
    y_ = y_[..., None] * div_term                       # (H, W, E/2)
    p_y = tc.cat([y_.sin(), y_.cos()], dim=-1)          # (H, W, E)
    
    # Combine x and y positional encodings
    # p = tc.cat([p_x, p_y], dim=-1)                    # (H, W, 2E)
    p = p_x + p_y
    p = p.permute(2, 0, 1)                              # (2E, H, W)
    return p

    
def DCT_Embedding_2D_(H, W, N_2, Plot=False):
    """
    Generate and visualize 2D DCT basis functions using SciPy.

    Args:
        N (int): Size of the DCT basis (NxN).
    """
    
    N = int(N_2**0.5)
    
    assert N == N_2**0.5
    
    p = np.zeros((N, N, H, W), dtype=np.float32)        # (N, N, H, W)

    for k1 in range(min(H, N)):
        for k2 in range(min(W, N)):
            # Generate 1D DCT basis vectors
            v1 = fft.dct(np.eye(H)[k1], norm='ortho')   # (H,)
            v2 = fft.dct(np.eye(W)[k2], norm='ortho')   # (W,)

            # Compute the 2D DCT basis as an outer product
            p[k1, k2] = np.outer(v1, v2)                # (H, W)

    # Visualize the 2D DCT basis functions
    if Plot:
        fig, axes = plt.subplots(N, N, figsize=(10, 10))
        for k1 in range(N):
            for k2 in range(N):
                axes[k1, k2].imshow(p[k1, k2], cmap='viridis')
                axes[k1, k2].axis('off')
        plt.suptitle("2D DCT Basis Functions using SciPy")
        plt.show()
    
    p = p.reshape(N_2, H, W)                            # (N^2, H, W)
    p -= p.mean(axis=(-1, -2))[:, None, None]           # (N^2, 1, 1)    
    p /= p.std(axis=(-1, -2))[:, None, None]            # (N^2, 1, 1)
    p = tc.tensor(p)                                    # (N^2, H, W)
    
    return p


class Pad_(nn.Module):
    def __init__(self, pad, mode='replicate'):
        super(Pad_, self).__init__()
        self.pad = pad  # (left, right, top, bottom, front, back)
        self.mode = mode

    def forward(self, x):
        return F.pad(x, self.pad, mode=self.mode)


###############################################################################
# Building Blocks

class Causal_Conv_3D_(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=(4, 4, 4), stride=(2, 2, 2)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # No internal padding — we pad manually
        self.conv = nn.Conv3d(
            C_in, C_out, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=0
        )
        
        # Calculate how much to pad on each axis
        self.t_pad = self.kernel_size[0] - 1           # causal (left-only)
        self.h_pad = (self.kernel_size[1] - 1) // 2    # symmetric
        self.w_pad = (self.kernel_size[2] - 1) // 2    # symmetric
        return

    def forward(self, x):
        # x: (B, C, T, H, W)
        
        t_pad, h_pad, w_pad = self.t_pad, self.h_pad, self.w_pad

        # Pad in (W_left, W_right, H_top, H_bottom, T_front, T_back)
        x = F.pad(x, (w_pad, w_pad, h_pad, h_pad, t_pad, 0), mode='replicate')

        return self.conv(x)


class Residual_Linear_Block_(nn.Module):
    def __init__(self, size:int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: tc.tensor):
        return x + self.act(self.ff(x))


class Residual_Conv_Block_(nn.Module):
    def __init__(self, c_in, c_out, E_t=None):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.E_t = E_t
        
        if E_t is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(E_t, c_in),
                nn.SiLU(),
                nn.Linear(c_in, c_in),
            )

        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            # nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            # nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(),
        )

        # self.use_attention = use_attention
        # if use_attention:
            # self.attention = nn.MultiheadAttention(embed_dim=c_out, num_heads=4, batch_first=True)


        return

    def forward(self, x, p_t=None):
        # x:   (B, c, h, w)
        # p_t: (B, E_t)
        
        if self.E_t is not None:
            p_t = self.time_mlp(p_t)[:, :, None, None]      # (B, c_in, 1, 1)
            x = x + p_t                                     # (B, c_in, h, w)
            
        x = self.block1(x)                                  # (B, c_out, h, w)      
        x = self.block2(x)                                  # (B, c_out, h, w)  
        
        # if self.use_attention:
        #     x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        #     x, _ = self.attention(x, x, x)
        #     x = x.permute(0, 2, 1).view(b, c, h, w)
        
        return x
    

class GCN_Block_(nn.Module):
    def __init__(self, C1:int=64, C2:int=64, use_res=False):
        super().__init__()
        
        self.C1 = C1
        self.C2 = C2
        self.Linear = nn.Linear(C1, C2)
        self.use_res = use_res
        
        if use_res:
            self.Residual = nn.Linear(C1, C2)

        return
    
    def forward(self, x: tc.Tensor, e_: tc.Tensor):#, b_: tc.Tensor):
        """
        Args:
            x:   (BN, C1)        node features
            e_:  (2, E)          edge index [src, dst]
            b_:  (BN,)           batch index for each node
        Returns:
            y:   (BN, C2)        updated node features
        """
        BN, C1 = x.shape
        E = e_.shape[1]

        row, col = e_  # (E,) x2
        
        # Step 1: Degree normalization
        deg = tc.bincount(row, minlength=BN).float()            # (B.N,)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]            # (E,)

        # Step 2: Linear transformation
        h = self.Linear(x)                                      # (B.N, C2)

        # Step 3: Optional residual
        if self.use_res:
            y = self.Residual(x)                                # (B.N, C2)
        else:
            y = tc.zeros_like(h)                                # (B.N, C2)

        # Step 4: Message passing (vectorized)
        messages = norm[..., None] * h[row]                     # (E, C2)
        y = y.index_add(0, col, messages)                       # (B.N, C2)
        return y


class Diff_Pool_(nn.Module):
    def __init__(self, C1:int=64, C2:int=64, M:int=16, use_embed=False):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.M = M
        self.use_embed = use_embed
        
        self.assign = nn.Linear(C1, M)
        self.embed = nn.Linear(C1, C2-3)
        return

    def forward(self, x, e_, b_):
        """
        Args:
            x:      (B.N, C1)  Input node features (float)
            e_:     (2, B.E1)  Edge connections (int)
            b_:     (B.N,)     Batch index for each node (int)

        Returns:
            x_:     (B.M, C2)  Pooled features
            e_:     (2, B.E2)  Pooled adjacency
            b_:     (B.M,)     Batch index for each node (int)

        """
        # x:  (B.N, C1) float
        # e_: (2, B.E1) int
        # b_: (B.N)     int
        
        BN, C1 = x.shape
        BE = e_.shape[1]
        B = b_.max().item() + 1
        M = self.M
        C2 = self.C2
        dev = x.device
        # print(x.shape, e_.shape)
        
        # if self.use_embed:
        #     # 1: Compute assignment scores
        #     S_ = tc.softmax(self.assign(x), dim=-1)             # (B.N, M)

        #     # 2: Embed original features
        #     Z_ = self.embed(x)                                  # (B.N, C2)
        # else:
        #     # S_ = tc.ones((BN, M)) / M                           # (B.N, M)
        #     S_ = tc.softmax(self.assign(x), dim=-1)             # (B.N, M)
            
        #     Z_ = x #@ tc.ones((C1, C2))                         # (B.N, C2)
            

        # Input: 
        # x = [Z_p | rest], shape: (B.N, C1), b_: (B.N,), M: clusters per batch
        
        # 1. Separate position and features
        p_ = x[:, :3]                                           # (B.N, 3)
        f_ = self.embed(x)                                      # (B.N, C2 - 3)
        
        # 2. Assignment matrix
        S_ = tc.softmax(self.assign(x), dim=-1)                 # (B.N, M)

        # 3. Flatten assignments and embeddings
        S_ = S_.view(-1)[:, None]                               # (B.N*M, 1)
        f_ = f_[:, None].expand(-1, M, -1).reshape(-1, C2-3)    # (B.N*M, C2 - 3)
        p_ = p_[:, None].expand(-1, M, -1).reshape(-1, 3)       # (B.N*M, 3)

        # 4. Compute global supernode indices for index_add
        I_ = (b_[:, None] * M + tc.arange(M, device=dev)[None, :]).flatten()  # (B.N * M,)
        
        # 5. Compute pooled features (unnormalized)
        Z_f = tc.zeros((B * M, C2-3), device=dev)               # (B*M, C2 - 3)
        Z_f = Z_f.index_add(0, I_, S_ * f_)                     # (B*M, C2 - 3)
        
        # 6. Compute pooled position (normalized)
        Z_p = tc.zeros((B * M, 3), device=dev)                  # (B*M, 3)
        Z_p = Z_p.index_add(0, I_, S_ * p_)
        
        # Compute normalizer per supernode (sum of assignment weights per supernode)
        w = tc.zeros(B * M, device=dev)                         # (B*M,)
        w = w.index_add(0, I_, S_[:, 0])                        # (B*M,)
        w = w.clamp_min(1e-10)                                  # (B*M,)

        # Normalize position only
        Z_p = Z_p / w[:, None]                                  # (B*M, 3)

        # 7. Final supernode features: [position | features]
        x = tc.cat([Z_p, Z_f], dim=-1)                          # (B*M, C2)
                
        # 5. Compute Pooled Adjacency Matrix
        S_ = S_.view(BN, M)                                     # (B.N, M)
        A_NN = tc.sparse_coo_tensor(e_, tc.ones(BE, device=dev), (BN, BN))  # (B.N, B.N)
        AS_ = tc.sparse.mm(A_NN, S_)                            # (B.N, M)
        ST_ = S_.T[None].repeat(B, 1, 1)                        # (B, M, B.N)
        ST_ *= F.one_hot(b_, num_classes=B).T.float()[:, None]  # (B, M, B.N)
        A_MM = ST_ @ AS_                                        # (B*M, M)
        A_MM = A_MM.view(B, M, M)                               # (B, M, M)
        
        # 6. Find k strongest connections
        k = 4
        src = tc.arange(M, device=dev).view(1, M, 1).repeat(B, 1, k)  # (B, M, k)        
        dst = A_MM.topk(k, dim=-1)[1]                           # (B, M, k)

        offset = tc.arange(B, device=dev).view(B, 1, 1)*M       # (B, 1, 1)
        src += offset                                           # (B, M, k)
        dst += offset                                           # (B, M, k)
        
        e_ = tc.stack([src.flatten(), dst.flatten()], dim=0)    # (2, B*M*k)
        
        b_ = tc.arange(B).repeat_interleave(M)                  # (B*M,)
        
        # for i in range(B):
            # plt.imshow(A_MM[i].detach().numpy()), plt.show()
        
        return x, e_, b_
        

class Cluster_Pool_(nn.Module):
    def __init__(self, M: int = 16, temperature: float = 1.0, Method='KMeans'):
        super().__init__()
        self.M = M
        self.temperature = temperature
        self.Method = Method

    def forward(self, x: tc.Tensor, e_: tc.Tensor, b_: tc.Tensor):
        """
        Args:
            x:   (B.N, D) - input point cloud
            e_:  (2, E)   - edge indices (unused here)
            b_:  (B.N,)   - batch index for each node
        Returns:
            x_pooled: (B.M, D)
        """
        dev = x.device
        M, temp = self.M, self.temperature
        B = b_.max().item() + 1
        BN, C = x.shape
        BE = e_.shape[1]
        D = 3

        # Step 1: FPS on each batch (returns (B, M, D))
        if self.Method == 'KMeans':
            centroids = KMeans_(x[:, :D], M, b_, iters=20)       # (B, M, D)
        elif self.Method == 'FPS':
            centroids = FPS_(x[:, :D], M, b_)                   # (B, M, D)

        # Step 2: Build centroids index tensor for each point
        # Expand centroids to match x: (B.N, M, D)
        c_ = centroids[b_]                                      # (B.N, M, D)

        # Step 3: Pairwise distance between each point and each centroid in its batch
        d_ = ((x[:, None, :D] - c_) ** 2).sum(dim=-1)           # (B.N, M)

        # Step 4: Soft assignment (closer centroid = higher score)
        S_ = F.softmax(-d_ / temp, dim=1).view(-1, 1)           # (B.N*M, 1)
        
        # x (B.N, C)
        f_ = x[:, 3:]                                           # (B.N, C-D)
        f_ = f_[:, None].expand(-1, M, -1).reshape(-1, C-D)     # (B.N*M, C-D)
        f_ *= S_                                                # (B.N*M, C-D)

        I_ = (b_[:, None] * M + tc.arange(M, device=dev)[None, :]).flatten()  # (B.N * M,)
        
        # # 6. Compute pooled position (normalized)
        x_ = tc.zeros((B * M, C-D), device=dev)                 # (B*M, C-D)
        x_ = x_.index_add(0, I_, f_)                            # (B*M, C-D) 
        x_ = tc.cat([centroids.reshape(-1, D), x_], dim=1)      # (B*M, C)
        
        # 7. Compute Pooled Adjacency Matrix
        S_ = S_.view(BN, M)                                     # (B.N, M)
        A_NN = tc.sparse_coo_tensor(e_, tc.ones(BE, device=dev), (BN, BN))  # (B.N, B.N)
        AS_ = tc.sparse.mm(A_NN, S_)                            # (B.N, M)
        ST_ = S_.T[None].repeat(B, 1, 1)                        # (B, M, B.N)
        ST_ *= F.one_hot(b_, num_classes=B).T.float()[:, None]  # (B, M, B.N)
        A_MM = ST_ @ AS_                                        # (B*M, M)
        A_MM = A_MM.view(B, M, M)                               # (B, M, M)
        
        # 6. Find k strongest connections
        k = 4
        src = tc.arange(M, device=dev).view(1, M, 1).repeat(B, 1, k)  # (B, M, k)        
        dst = A_MM.topk(k, dim=-1)[1]                           # (B, M, k)

        offset = tc.arange(B, device=dev).view(B, 1, 1)*M       # (B, 1, 1)
        src += offset                                           # (B, M, k)
        dst += offset                                           # (B, M, k)
        
        e_ = tc.stack([src.flatten(), dst.flatten()], dim=0)    # (2, B*M*k)
        
        b_ = tc.arange(B).repeat_interleave(M)                  # (B*M,)

        # tc.empty(2, 0, dtype=tc.long, device=dev)
        return x_, e_, b_


###############################################################################
# Standard Networks

class NN_(nn.Module):
    def __init__(self, D_in:tuple, D_out:tuple, h=256, l=10, Act='GELU', Output_Act=None):
        super().__init__()
        
        self.D_in = D_in
        self.D_out = D_out
        
        
        self.d_in = np.prod(D_in)
        self.d_out = np.prod(D_out)
        self.h = h
        self.l = l
        
        Activation = Activation_Functions[Act]
        
        layers = [nn.Flatten(), nn.Linear(self.d_in, self.h), Activation]
        
        for _ in range(self.l):
            layers.append(Residual_Linear_Block_(self.h))
        
        layers.append(nn.Linear(self.h, self.d_out))
        
        if Output_Act is not None:
            Output_Act = Activation_Functions[Output_Act]
            layers.append(Output_Act)
        
        layers.append(nn.Unflatten(1, D_out))
        self.layers = nn.Sequential(*layers)
        
        return
    
    def forward(self, x):
        # x: (B, D_in)
        
        B, D = x.shape[0], x.shape[1:]

        y = self.layers(x)                                    # (B, D_out)  

        return y


class ERNN__(nn.Module):
    
    def __init__(self, D_in:int, D_out:int, E:int):
        super().__init__()
        
        self.D_in = D_in
        self.D_out = D_out     
        self.E = E
        
        hidden_size = 128
        hidden_layers = 10
        layers = [nn.Linear(D_in, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Residual_Linear_Block_(hidden_size))
        layers.append(nn.Linear(hidden_size, D_out))
        self.layers = nn.Sequential(*layers)
        
        return
    
    def forward(self, x, t):
        
        p_t = Sinusoidal_Embedding_1D_(t, E=self.E)               # (B, E)
        p_x = Sinusoidal_Embedding_1D_(x[:, 0], E=self.E)         # (B, E) 
        p_y = Sinusoidal_Embedding_1D_(x[:, 1], E=self.E)         # (B, E) 
        
        x_p = tc.cat([p_t, p_x, p_y], axis=-1)                    # (B, D_in)  
        
        y = self.layers(x_p)                                      # (B, D_out)          
        return y
        
    
class DDNN__(nn.Module):
    
    def __init__(self, D_in:tuple, D_out:tuple, E:int):
        super().__init__()
        
        self.D_in = D_in
        self.D_out = D_out
        self.E = E
        
        self.d_in = np.prod(D_in)
        self.d_out = np.prod(D_out)
        
        hidden_size = 256
        hidden_layers = 10
        layers = [nn.Flatten(), nn.Linear(self.d_in, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Residual_Linear_Block_(hidden_size))
        layers.append(nn.Linear(hidden_size, self.d_out))
        # layers.append(nn.GELU())
        layers.append(nn.Unflatten(1, D_out))
        self.layers = nn.Sequential(*layers)
        
        return
    
    def forward(self, x, t):
        # x (B, D...)
        # t (B,)
        
        B = t.shape[0]       
        D = x.shape[1:]
        
        x = x.view(B, -1)
        
        p_t = Sinusoidal_Embedding_1D_(t, E=self.E)             # (B, E)
        x_p = tc.cat([x, p_t], axis=-1)                         # (B, D_in)  

        y = self.layers(x_p)                                    # (B, D_out)  

        return y
    
###############################################################################
# Convolutional Networks

class Image_AE_(nn.Module):
    def __init__(self, C=3, H=128, W=128, c=256, E=8, Clamp=True, Activation='GELU'):
        super(Image_AE_, self).__init__()
        # x: (B, C, H, W)
        self.C = C
        self.H = H
        self.W = W
        self.E = E
        
        self.f_s = 8
        # h = H//f_s
        # w = W//f_s
        self.c = c
        self.h = h = H//self.f_s
        self.w = w = W//self.f_s
        
        self.Clamp = Clamp
        
        Activation = Activation_Functions[Activation]
        self.Activation = Activation
        
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d(C+E, 32, kernel_size=4, stride=2, padding=1),             # (B, 32, H/2, W/2)
            Activation,
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),              # (B, 32, H/4, W/4)
            Activation,
            nn.Conv2d(32, c, kernel_size=4, stride=2, padding=1),               # (B, 32, H/8, W/8)                           
            Activation,
            nn.Flatten(),                                                       # (B, 10x4x4)   
            nn.Linear(c*h*w, c*h*w),                                            # (B, 10x4x4)
            Activation,

            # nn.Tanh(),
            # nn.Unflatten(1, (c, h, w)),                                       # (B, c, H/8, W/8)
            nn.Linear(c*h*w, 10),                                               # (B, 10)
            nn.Softmax(),
            # Softmax_Temperature_(0.5),
        )
        
        # Decoder
        self.Decoder = nn.Sequential(
            
            nn.Linear(10, c*h*w),                                               # (B, 10)
            # nn.Linear(c*h*w, c*h*w),                                            # (B, c*h*w)
            nn.Unflatten(1, (c, h, w)),                                         # (B, c, H/8, W/8)
            
            nn.ConvTranspose2d(c, 32, kernel_size=4, stride=2, padding=1),      # (B, 32, H/4, W/4)
            Activation,
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),     # (B, 32, H/2, W/2)
            Activation,
            nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),      # (B, C, H, W)
            Activation,
        )

        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # p = Grid_Position_Embedding_(H//8, W//8).permute(2, 0, 1)[None, ...].repeat(B, 1, 1, 1)         # (B, 2, H, W)
        # p = Sinusoidal_Embedding_2D_(H, W, self.E).permute(2, 0, 1)[None, ...].repeat(B, 1, 1, 1)     # (B, E, H, W)
        # x = tc.cat([x, p], axis=1)
        
        # print(p.shape)
        # sys.exit()
        
        z_ = self.Encoder(x)    # (B, c, h, w)
        x_ = self.Decoder(z_)   # (B, C, H, W)
        
        # plt.imshow(z_[:10].detach().numpy()), plt.show()
        
        if self.Clamp:
            x_ = tc.clamp(x_, min=0.0, max=1.0)
        
        return x_, z_


class Image_AE2_(nn.Module):
    def __init__(self, C=3, H=128, W=128, c=256, E=8, Clamp=True):
        super(Image_AE2_, self).__init__()
        # x: (B, C, H, W)
        self.C = C
        self.H = H
        self.W = W
        self.E = E
        
        self.f_s = 8
        # h = H//f_s
        # w = W//f_s
        self.c = c
        self.h = h = H//self.f_s
        self.w = w = W//self.f_s
        
        self.Clamp = Clamp
        Activation = Activation_Functions['SiLU']
        
        class Block_(nn.Module):
            def __init__(self, C_in, C_out, H, W, Type='D'):
                super().__init__()
                # x: (B, C_in, H, W)

                if Type == 'D':
                    self.Conv = nn.Sequential(
                        nn.Conv2d(C_in, C_out, kernel_size=4, stride=2, padding=1),             # (B, C_out, H/2, W/2)
                    )
                    
                    # self.Linear = nn.Sequential(
                    #     nn.Flatten(),                                               # (B, C_in*H*W)
                    #     nn.Linear(C_in*H*W, C_out*H*W//4),                          # (B, C_out*H*W/4)
                    #     nn.Unflatten(1, (C_out, H//2, W//2))                        # (B, C_out, H/2, W/2)
                    # )

                elif Type == 'U':
                    self.Conv = nn.Sequential(
                        nn.ConvTranspose2d(C_in, C_out, kernel_size=4, stride=2, padding=1),    # (B, C_out, 2H, 2W)
                    )
                    
                    # self.Linear = nn.Sequential(
                    #     nn.Flatten(),                                               # (B, C_in*H*W)
                    #     nn.Linear(C_in*H*W, C_out*H*W*4),                           # (B, C_out*H*W*4)
                    #     nn.Unflatten(1, (C_out, H*2, W*2))                          # (B, C_out, 2H, 2W)
                    # )

                return

            def forward(self, x):
                # x:   (B, C_in, H, W)
                
                # print(x.shape)
                x1 = self.Conv(x)        # (B, C_out, H/2, W/2) or (B, C_out, 2H, 2W) 
                # x2 = self.Linear(x)      # (B, C_out, H/2, W/2) or (B, C_out, 2H, 2W)
                # y = x1 + x2              # (B, C_out, H/2, W/2) or (B, C_out, 2H, 2W)
                y = x1
                # print(y.shape)
                return y
            
        # Encoder
        self.Encoder = nn.Sequential(
            Block_(C_in=C+E, C_out=c, H=H, W=W, Type='D'),                # (B, c, H/2, W/2)
            Activation,
            Block_(C_in=c, C_out=c, H=H//2, W=W//2, Type='D'),          # (B, c, H/4, W/4)
            Activation,
            Block_(C_in=c, C_out=c, H=H//4, W=W//4, Type='D'),          # (B, c, H/8, W/8)
        )
        
        # Decoder
        self.Decoder = nn.Sequential(
            Block_(C_in=c, C_out=c, H=H//8, W=W//8, Type='U'),          # (B, c, H/4, W/4)
            Activation,
            Block_(C_in=c, C_out=c, H=H//4, W=W//4, Type='U'),          # (B, c, H/2, W/2)
            Activation,
            Block_(C_in=c, C_out=C, H=H//2, W=W//2, Type='U'),          # (B, C, H, W)
            Activation,
        )
        
        B = 10
        self.p = Grid_Position_Embedding_(H, W)                       # (2, H, W)
        # self.p = Sinusoidal_Embedding_2D_(H, W, self.E)               # (E, H, W)
        # self.p = DCT_Embedding_2D_(H, W, self.E)                        # (E, H, W)
        self.expanded = False
        
    def forward(self, x, c=None):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        if not self.expanded:
            self.p = self.p[None].expand(B, -1, -1, -1)                     # (B, E, H, W)
            self.expanded = True
            
        # print(x.shape, self.p.shape)
        # print(x.dtype)

        # x += tc.rand_like(x) * 0.1
        # x = (x[:, :, None] * self.p[:, None]).reshape(B, C*self.E, H, W)     # (B, CE, H, W)
        x = tc.cat([x, self.p], axis=1)      # (B, C+E, H, W)
        # plt.imshow(p[0, 1, :, :])
        
        # print(p.shape)
        # print(x.shape)
        # sys.exit()
        
        z_ = self.Encoder(x)    # (B, c, h, w)
        x_ = self.Decoder(z_)   # (B, C, H, W)
        
        # plt.imshow(z_[:10].detach().numpy()), plt.show()
        
        if self.Clamp:
            x_ = tc.clamp(x_, min=0.0, max=1.0)
        
        return x_, z_


class UNET_(nn.Module):
    def __init__(self, C_in=3, C_out=3, c=64, E_t=None):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.c = c
        self.E_t = E_t
        
        # if E_t != 0:
            # self.time_embedding = SinusoidalTimeEmbedding(E_t)  # (E_t)
            # print(self.time_embedding.shape)
        
        # Encoder
        self.down1 = Residual_Conv_Block_(C_in, c,      E_t)
        self.down2 = Residual_Conv_Block_(c, c * 2,     E_t)
        self.down3 = Residual_Conv_Block_(c * 2, c * 4, E_t)
        self.down4 = Residual_Conv_Block_(c * 4, c * 8, E_t)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = Residual_Conv_Block_(c * 8, c * 8, E_t)
        
        # Decoder
        self.up4 = Residual_Conv_Block_(c * 8, c * 4,   E_t)
        self.up3 = Residual_Conv_Block_(c * 4, c * 2,   E_t)
        self.up2 = Residual_Conv_Block_(c * 2, c,       E_t)
        self.up1 = Residual_Conv_Block_(c, C_out,       E_t)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        return   
    
    def forward(self, x, t=None):
        # x: (B, C, H, W)
        # t: (B,)
        
        # Time Embedding
        if self.E_t is not None:
            # p_t = self.time_embedding(t)                  # (B, E_t)
            p_t = Sinusoidal_Embedding_1D_(t, self.E_t)     # (B, E_t)
        else:
            p_t = None
            
        # Encoder
        x1 = self.down1(x, p_t)                             # (B, C_in, H, W)
        x2 = self.down2(self.pool(x1), p_t)                 # (B, 2c, H/2, W/2)
        x3 = self.down3(self.pool(x2), p_t)                 # (B, 4c, H/4, W/4)
        x4 = self.down4(self.pool(x3), p_t)                 # (B, 8c, H/8, W/8)
        
        # Bottleneck
        z_ = self.bottleneck(self.pool(x4), p_t)            # (B, 16c, H/16, W/16)
        
        # Decoder
        x4 = self.up4(self.upsample(z_) + x4, p_t)          # (B, 8c, H/8, W/8)
        x3 = self.up3(self.upsample(x4) + x3, p_t)          # (B, 4c, H/4, W/4)
        x2 = self.up2(self.upsample(x3) + x2, p_t)          # (B, 2c, H/2, W/2)
        x1 = self.up1(self.upsample(x2) + x1, p_t)          # (B, C_out, H, W)
        
        return x1
   
    
class ViT_AE_(nn.Module):
    def __init__(self, C=3, H=128, W=128, c=256, num_heads=8, num_layers=6, patch_size=16, latent_dim=512, Activation='GELU'):
        super(ViT_AE_, self).__init__()

        self.C = C
        self.H = H
        self.W = W
        self.c = c
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.f_s = patch_size
        self.h = h = H // self.f_s
        self.w = w = W // self.f_s
        
        self.Activation = Activation_Functions[Activation]

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(C, c, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(tc.randn(1, self.h * self.w, c))                  # (1, hw, c)
        # self.pos_embed = DCT_Embedding_2D_(h, w, c, Plot=True).reshape(1, c, h*w).transpose(1, 2) # (1, hw, c)

        # Stacking Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=c, nhead=num_heads, dim_feedforward=c*4, batch_first=True)
            for _ in range(num_layers)])
        
        # Linear layer to produce latent representation from transformer output
        self.encode_latent = nn.Linear(c, latent_dim)


        # Stacking Transformer Decoder Layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=c, nhead=num_heads, dim_feedforward=c*4, batch_first=True)
            for _ in range(num_layers)])
        
        self.decode_latent = nn.Linear(latent_dim, c)

        # Decoder: Upsampling + Deconvolution (transpose convolution layers)
        self.decode_conv = nn.ConvTranspose2d(c, C, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        # X: (B, C, H, W)
        B, C, H, W = x.shape
        c, h, w = self.c, self.h, self.w

        # Encoder: Patch embedding + Positional Embedding
        x = self.patch_embed(x)                     # (B, c, hw)
        x = x.flatten(2).transpose(1, 2)            # (B, hw, c)
        x = x + self.pos_embed                      # (B, hw, c)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)                            # (B, hw, c)

        # Bottleneck: Latent representation (taking the first token from transformer output)
        x = x.mean(dim=1)                           # (B, c)
        z_ = self.encode_latent(x)                  # (B, d)
        
        # Decoder: Reconstructing the image from latent
        x_ = self.decode_latent(z_)                 # (B, c)
        x_ = x_[:, None].expand(-1, h*w, -1)        # (B, hw, c)
        
        # Apply transformer decoder layers
        for layer in self.decoder_layers:
            x_ = layer(x_)                          # (B, hw, c)
        
        # Final layer to get back to the original image size
        x_ = x_.transpose(1, 2).view(B, c, h, w)    # (B, c, h, w)
        x_ = self.decode_conv(x_)                 # (B, C, H, W)

        # print(x.shape, z_.shape, x_.shape)
        return x_, z_

###############################################################################

class Video_AE_(nn.Module):
    def __init__(self, c=64, C=3, L=64, H=64, W=64, f_s=8, f_t=4):
        super(Video_AE_, self).__init__()
        # x: (B, L, C, H, W)
        self.c = c          # Latent Channels
        self.L = L          # Clip Length
        self.C = C          # Channels (RGB = 3)
        self.H = H          # Height
        self.W = W          # Width
        self.f_s = f_s      # Spatial Downsampling
        self.f_t = f_t      # Temporal Downsampling
        self.E = 2          # Position Embedding
        
        # c = 32
        # l = L//f_t
        # h = H//f_s
        # w = W//f_s        
        
        Activation = Activation_Functions['SiLU']
        
        # print(c, L, H, W, c*L*H*W//2**12)
        # sys.exit()
        
        # Encoder
        self.Encoder = nn.Sequential(
            Causal_Conv_3D_(C+self.E, c),       # (B, c, L/2, H/2, W/2)
            Activation,
            Causal_Conv_3D_(c, c),              # (B, c, L/4, H/4, W/4)
            Activation,
            Causal_Conv_3D_(c, c),              # (B, c, L/8, H/8, W/8)
            # Activation,
            # Causal_Conv_3D_(c, c),              # (B, c, L/16, H/16, W/16)
            
            # nn.Flatten(),
            # nn.Linear(c*L*H*W//2**9, c*L*H*W//2**12),
            # nn.Unflatten(1, (c, L//16, H//16, W//16)),

            nn.Tanh(),


        )
        
        # Decoder
        self.Decoder = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(c*L*H*W//2**12, c*L*H*W//2**9),
            # nn.Unflatten(1, (c, L//8, H//8, W//8)),

            # nn.ConvTranspose3d(c, c, kernel_size=4, stride=2, padding=1),   # (B, c, L/8, H/8, W/8)
            # Activation,
            nn.ConvTranspose3d(c, c, kernel_size=4, stride=2, padding=1),   # (B, c, L/4, H/4, W/4)
            Activation,
            nn.ConvTranspose3d(c, c, kernel_size=4, stride=2, padding=1),   # (B, c, L/2, H/2, W/2)
            Activation,
            nn.ConvTranspose3d(c, C, kernel_size=4, stride=2, padding=1),   # (B, C, L, H, W)
        )
        
        self.p = Grid_Position_Embedding_(H, W)                             # (2, H, W)
        self.expanded = False
        return

    def forward(self, x, c=None):
        # Input x: (B, C, L, H, W)
        B, C, L, H, W = x.shape
        
        # Inject Positional information
        if not self.expanded:
            self.p = self.p[None, :, None].expand(B, -1, L, -1, -1)     # (B, E, L, H, W)
            self.expanded = True
        
        
        x = tc.cat([x, self.p], axis=1)                     # (B, C+E, L, H, W)
        
        # print(x.shape)
        z_ = self.Encoder(x)   # (B, c, l, h, w)
        # print(z_.shape)
        x_ = self.Decoder(z_)  # (B, C, L, H, W)
        # print(x_.shape)

        
        # print(x.shape, z_.shape, x_.shape)
        # sys.exit()
        
        x_ = tc.clamp(x_, min=0.0, max=1.0)
        
        return x_, z_
    
###############################################################################

class ARCDM_(nn.Module):
    
    def __init__(self, D_in:tuple, D_out:tuple, E:int):
        super().__init__()
        
        self.D_in = D_in
        self.D_out = D_out
        self.E = E
        
        self.d_in = np.prod(D_in)
        self.d_out = np.prod(D_out)
        
        hidden_size = 1024
        hidden_layers = 10
        layers = [nn.Flatten(), nn.Linear(self.d_in, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Residual_Linear_Block_(hidden_size))
        layers.append(nn.Linear(hidden_size, self.d_out))
        # layers.append(nn.GELU())
        layers.append(nn.Unflatten(1, D_out))
        self.layers = nn.Sequential(*layers)
        
        return
    
    # def forward(self, x, c=None):
    def forward(self, x, t, c=None):
        # x     (B, D...)
        # t     (B,)
        # c     (B, L-1, D...)
        assert x.shape[0] == t.shape[0], f"{x.shape}, {t.shape}"
        B = t.shape[0]
        D = x.shape[1:]
        
        # print(x.shape, t.shape, c.shape)
        
        if c is None:
            c = tc.zeros((B, 3,) + D)
        
        x = tc.cat([x[:, None], c], axis=1)         # (B, l, D)
        x = x.view(B, -1)                           # (B, l*D...)
        
        # print(x.shape)
        
        p_t = Sinusoidal_Embedding_1D_(t, E=self.E)             # (B, E)
        x_p = tc.cat([x, p_t], axis=-1)                         # (B, l*D+E) 
        
        # print(x_p.shape)

        y = self.layers(x_p)                                    # (B, D_out)  
        # y = self.layers(x)                          # (B, D_out)  

        # print(y.shape)

        return y

###############################################################################
# Point Cloud Network

class PC_AE_(nn.Module):
    def __init__(self):
        super(PC_AE_, self).__init__()
        # x: (B, N, D)
        N = 1067
        D = 3
        C = 1024
        
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Flatten(),               # (B, ND)
            nn.Linear(N*D, C),          # (B, C)
            nn.Tanh(),
            nn.Linear(C, C//2),         # (B, C/2)
            nn.Tanh(),
            nn.Linear(C//2, C//4),      # (B, C/4)
            nn.Tanh(),
            nn.Linear(C//4, C//16),         # (B, 1)
            # nn.Tanh(),
            # nn.Linear(C, 10),         # (B, 1)
        
        )
        
        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(C//16, C//4),         # (B, C/4)
            nn.Tanh(),
            nn.Linear(C//4, C//2),       # (B, C/2)
            nn.Tanh(),
            nn.Linear(C//2, C),         # (B, C)
            nn.Tanh(),
            nn.Linear(C, N*D),            # (B, ND)
            # nn.Linear(10, N*D),         # (B, 1)
            nn.Unflatten(1, (N, D))     # (B, N, D)
        )

        
    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        
        z_ = self.Encoder(x)    # (B, 1)
        x_ = self.Decoder(z_)   # (B, N, D)
        
        return x_, z_


class PointNet_AE_(nn.Module):
    def __init__(self, D: int = 3, k: int = 5, c: int = 64, M: int = 1000):
        super(PointNet_AE_, self).__init__()
        self.D = D  # Point Cloud Dimensions
        self.k = k  # Neighbourhood Size
        self.c = c  # Feature Map Dimensions
        self.M = M  # Output Point Cloud Size
        
        Activation = nn.GELU()
        
        self.Encoder = nn.Sequential(
            nn.Linear(D, c),                    # (BNk, c)
            Activation,
            nn.Linear(c, c),                    # (BNk, c)
            Activation,
            nn.Linear(c, c),                    # (BNk, c)
            Activation,
            )

        self.Decoder = nn.Sequential(
            nn.Linear(c, c),                    # (B, c)
            Activation,
            nn.Linear(c, c),                    # (B, c)
            Activation,
            nn.Linear(c, self.M*D),             # (B, MD)
            nn.Unflatten(1, (self.M, D)),       # (B, M, D)
            )
        
        return
        
        
    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        
        x = KNN_(x, self.k)[0]                              # (B, N, k, D)
        
        # x: (B, N, k, D)
        # B, N, k, D = x.shape
        
        x = x.view(-1, self.D)                              # (BNk, D)
        
        z_ = self.Encoder(x)                                # (BNk, c)
        
        z_ = z_.view(B, N, self.k, self.c)                  # (B, N, k, c)
        z_ = z_.max(dim=2)[0]                               # (B, N, c)
        z_ = z_.max(dim=1)[0]                               # (B, c)
        
        x_ = self.Decoder(z_)                               # (B, M, D)
        
        # print(x.shape, z_.shape, x_.shape)
        return x_, z_


class GCN_AE_(nn.Module):
    def __init__(self, C:int, D:int=3, c:int=64):
        super(GCN_AE_, self).__init__()
        self.C = C  # Point Cloud Feature Dimensions
        self.D = D  # Point Cloud Dimensions
        self.c = c  # Feature Map Dimensions
        self.Act = Activation_Functions['GELU']
        
        M0 = 1000
        M1 = 256
        M2 = 64
        M3 = 16
        
        class Encoder_(nn.Module):
            def __init__(self, C:int, D:int=3, c:int=64, Act=self.Act):
                super(Encoder_, self).__init__()
                self.C = C
                self.D = D
                self.c = c
                self.Act = Act
                # Method = 'FPS'
                
                # self.gcn1 = GCN_Block_(C1=C, C2=c-D, use_res=True)
                # self.pool1 = Cluster_Pool_(M=256, Method=Method)
                
                # self.gcn2 = GCN_Block_(C1=c, C2=c*2-D, use_res=True)
                # self.pool2 = Cluster_Pool_(M=64, Method=Method)
                
                # self.gcn3 = GCN_Block_(C1=c*2, C2=c*4-D, use_res=True)
                # self.pool3 = Cluster_Pool_(M=16, Method=Method)
                
                self.gcn1 = GCN_Block_(C1=C, C2=c, use_res=True)
                self.pool1 = Diff_Pool_(C1=c, C2=c, M=M1)
                
                self.gcn2 = GCN_Block_(C1=c, C2=c*2, use_res=True)
                self.pool2 = Diff_Pool_(C1=c*2, C2=c*2, M=M2)
                
                self.gcn3 = GCN_Block_(C1=c*2, C2=c*4, use_res=True)
                self.pool3 = Diff_Pool_(C1=c*4, C2=c*4, M=M3)
                
                return

            def forward(self, x:tc.tensor, e_:tc.tensor, b_:tc.tensor):
                # x0:  (B.N, C)
                # e0: (2, B.E)
                # b0: (B.N,)
                BN, C = x.shape
                E = e_.shape[1]
                
                # Plot_Graph_(e0.T, X_=x0, B_=b0, i=0)

                # x1 = self.gcn1(x0, e0)                      # (B.N, c-D)
                # x1 = tc.cat([x0[:, :3], x1], dim=-1)        # (B.N, c)
                # Plot_Graph_(e0.T, X_=x1, B_=b0, i=0)
                
                # x1, e1, b1 = self.pool1(x1, e0, b0)         # (B.M1, c) (2, B.E2)                
                # Plot_Graph_(e1.T, X_=x1, B_=b1, i=0)
                # # x = self.Act(x)                           # (B.M1, c)

                # x2 = self.gcn2(x1, e1)                      # (B.M1, 2c-D)
                # x2 = tc.cat([x1[:, :3], x2], dim=-1)        # (B.M1, 2c)
                # Plot_Graph_(e1.T, X_=x2, B_=b1, i=0)

                # x2, e2, b2 = self.pool2(x2, e1, b1)         # (B.M2, 2c)
                # Plot_Graph_(e2.T, X_=x2, B_=b2, i=0)
                # # x = self.Act(x)                           # (B.M2, 2c)
                
                # x3 = self.gcn3(x2, e2)                      # (B.M2, 4c-D)
                # x3 = tc.cat([x2[:, :3], x3], dim=-1)        # (B.M2, 4c)
                # Plot_Graph_(e2.T, X_=x3, B_=b2, i=0)

                # x3, e3, b3 = self.pool3(x3, e2, b2)         # (B.M3, 4c)
                # Plot_Graph_(e3.T, X_=x3, B_=b3, i=0)
                # # x = self.Act(x)                           # (B.M3, 4c)
                
                # return x3, e3, b3
                

                x = self.gcn1(x, e_)                        # (B.N, c)
                x, e_, b_ = self.pool1(x, e_, b_)           # (B.M1, c) (2, B.E2)                
                # x = self.Act(x)                             # (B.M1, c)

                x = self.gcn2(x, e_)                        # (B.M1, 2c)
                x, e_, b_ = self.pool2(x, e_, b_)           # (B.M2, 2c)
                # x = self.Act(x)                             # (B.M2, 2c)
                
                x = self.gcn3(x, e_)                        # (B.M2, 4c)
                x, e_, b_ = self.pool3(x, e_, b_)           # (B.M3, 4c)
                # x = self.Act(x)                             # (B.M3, 4c)
                
                return x, e_, b_
            
        class Decoder_(nn.Module):
            def __init__(self, C:int, D:int=3, c:int=64, Act=self.Act):
                super(Decoder_, self).__init__()
                self.Act = Act
                Method = 'FPS'
                
                # self.gcn1 = GCN_Block_(C1=4*c, C2=4*c-D, use_res=True)
                # self.pool1 = Cluster_Pool_(M=64, Method=Method)
        
                # self.gcn2 = GCN_Block_(C1=c, C2=c*2-D, use_res=True)
                # self.pool2 = Cluster_Pool_(M=256, Method=Method)
                
                
                # self.gcn3 = GCN_Block_(C1=c*2, C2=c*4-D, use_res=True)
                # self.pool3 = Cluster_Pool_(M=1024, Method=Method)
                
                
                self.gcn1 = GCN_Block_(C1=c*4, C2=c*2, use_res=True)
                self.pool1 = Diff_Pool_(C1=c*2, C2=c*2, M=M2)
                
                self.gcn2 = GCN_Block_(C1=c*2, C2=c, use_res=True)
                self.pool2 = Diff_Pool_(C1=c, C2=c, M=M1)
                
                self.gcn3 = GCN_Block_(C1=c, C2=C, use_res=True)
                self.pool3 = Diff_Pool_(C1=C, C2=C, M=M0)
                
                self.head = nn.Linear(C, C)
                
                return
        
            def forward(self, x:tc.tensor, e_:tc.tensor, b_:tc.tensor):
                # x:  (B.M3, 4c)
                # e_: (2, B.E3)
                # b_: (B.M3,)
                BM, _ = x.shape
                E = e_.shape[1]
                
                # Plot_Graph_(e0.T, X_=x0, B_=b0, i=0)
        
                # x1 = self.gcn1(z, ez)                      # (B.M3, 2c-3)
                # print(x1.shape)
                # sys.exit()
                
                # x1 = tc.cat([x0[:, :3], x1], dim=-1)        # (B.N, c)
                # Plot_Graph_(e0.T, X_=x1, B_=b0, i=0)
                
                # x1, e1, b1 = self.pool1(x1, e0, b0)         # (B.M1, c) (2, B.E2)                
                # Plot_Graph_(e1.T, X_=x1, B_=b1, i=0)
                # # x = self.Act(x)                           # (B.M1, c)
        
                # x2 = self.gcn2(x1, e1)                      # (B.M1, 2c-3)
                # x2 = tc.cat([x1[:, :3], x2], dim=-1)        # (B.M1, 2c)
                # Plot_Graph_(e1.T, X_=x2, B_=b1, i=0)
        
                # x2, e2, b2 = self.pool2(x2, e1, b1)         # (B.M2, 2c)
                # Plot_Graph_(e2.T, X_=x2, B_=b2, i=0)
                # # x = self.Act(x)                           # (B.M2, 2c)
        
                # x3 = self.gcn3(x2, e2)                      # (B.M2, 4c-3)
                # x3 = tc.cat([x2[:, :3], x3], dim=-1)        # (B.M2, 4c)
                # Plot_Graph_(e2.T, X_=x3, B_=b2, i=0)
        
                # x3, e3, b3 = self.pool3(x3, e2, b2)         # (B.M3, 4c)
                # Plot_Graph_(e3.T, X_=x3, B_=b3, i=0)
                # # x = self.Act(x)                           # (B.M3, 4c)
                
                # return x3, e3, b3
            
                x = self.gcn1(x, e_)                        # (B.M3, 2c)
                x, e_, b_ = self.pool1(x, e_, b_)           # (B.M2, 2c) (2, B.E2)                
                # x = self.Act(x)                             # (B.M2, 2c)

                x = self.gcn2(x, e_)                        # (B.M2, c)
                x, e_, b_ = self.pool2(x, e_, b_)           # (B.M1, c)
                # x = self.Act(x)                             # (B.M1, c)
                
                x = self.gcn3(x, e_)                        # (B.M1, C)
                x, e_, b_ = self.pool3(x, e_, b_)           # (B.M0, C)
                
                x = self.head(x)                            # (B.M0, C)
                
                return x, e_, b_
                
        self.Encoder = Encoder_(C=C, D=D, c=c)
        self.Decoder = Decoder_(C=C, D=D, c=c)
        return
    
    def forward(self, x:tc.tensor, e_:tc.tensor):
        # x:  (B, N, C)
        # e_: (B, E, 2)
        B, N, C = x.shape
        E = e_.shape[1]

        # Plot_Graph_(e_[0], X_=x[0])

        # x0, e0, b0 = Batch_Graph_(x0, e0)           # (B.N, C)   (2, B.E0) (B.N,)
        # z, ez, bz = self.Encoder(x0, e0, b0)        # (B.M3, 4c) (2, B.Ez) (B.M3,)
        # x_, e_, b_ = self.Decoder(z, ez, bz)        # (B.M0, C)  (2, B.Ek) (B.M0)
        
        x, e_, b_ = Batch_Graph_(x, e_)                     # (B.N, C)   (2, B.E) (B.N,)
        
        z_, e_, b_ = self.Encoder(x, e_, b_)                # (B.M3, 4c) (2, B.E3) (B.M3,)
        
        x_, e_, b_ = self.Decoder(z_, e_, b_)               # (B.M0, C)  (2, B.E) (B.M0)

        x_ = tc.cat([x_[:, :3], 
                      tc.sigmoid(x_[:, 3:6]), 
                      x_[:, 6:]], dim=-1)                    # (B.M0, C)
        x_ = x_.view(B, -1, C)                              # (B, M0, C)
        
        # print(z_.shape, x_.shape, e_.shape, b_.shape)
        return x_, z_


class DGCNN_(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        # self.C = C  # Point Cloud Feature Dimensions
        # self.D = D  # Point Cloud Dimensions
        # self.c = c  # Feature Map Dimensions
        self.Act = Activation_Functions['GELU']
        
        # M0 = 1000
        # M1 = 256
        # M2 = 64
        # M3 = 16
        
        
        self.bn1 = nn.BatchNorm2d(1*c)
        self.bn2 = nn.BatchNorm2d(2*c)
        self.bn3 = nn.BatchNorm2d(4*c)
        self.bn4 = nn.BatchNorm2d(8*c)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(D+C, c, kernel_size=1, bias=False),
                                    self.bn1,
                                    self.Act)
        
        self.conv2 = nn.Sequential(nn.Conv2d(c, 2*c, kernel_size=1, bias=False),
                                    self.bn2,
                                    self.Act)
        
        self.conv3 = nn.Sequential(nn.Conv2d(2*c, 4*c, kernel_size=1, bias=False),
                                    self.bn3,
                                    self.Act)
        
        self.conv4 = nn.Sequential(nn.Conv2d(4*c, 8*c, kernel_size=1, bias=False),
                                    self.bn4,
                                    self.Act)
        
        self.conv5 = nn.Sequential(nn.Conv1d(8*c, 16*c, kernel_size=1, bias=False),
                                    self.bn5,
                                    self.Act)
        
        self.linear1 = nn.Linear(16*c, 8*c, bias=False)
        self.bn6 = nn.BatchNorm1d()
        # self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

###############################################################################


# def knn(x, k):
#     inner = -2*torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
#     return idx


# def get_graph_feature(x, k=20, idx=None):
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         idx = knn(x, k=k)   # (batch_size, num_points, k)
#     device = torch.device('cuda')

#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

#     idx = idx + idx_base

#     idx = idx.view(-1)
 
#     _, num_dims, _ = x.size()

#     x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size*num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims) 
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
#     feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
#     return feature


# class PointNet(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(PointNet, self).__init__()
#         self.args = args
#         self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
#         self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(args.emb_dims)
#         self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout()
#         self.linear2 = nn.Linear(512, output_channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.adaptive_max_pool1d(x, 1).squeeze()
#         x = F.relu(self.bn6(self.linear1(x)))
#         x = self.dp1(x)
#         x = self.linear2(x)
#         return x















# def Sinusoidal_Embedding_(x:tc.tensor, E:int=256):
#     # x (B, D)
#     B, D = x.shape
    
#     d_ = tc.arange(E, dtype=tc.float32)                # (E,)
#     k = tc.exp(-d_/E * tc.log(tc.tensor(10000.0)))     # (E,)
    
#     # print(x[..., None].shape, k[None, None, :].shape)
#     # sys.exit()
    
#     kx = k[None, None, :] * x[..., None]               # (B, D, E)
#     p = tc.zeros((B, D, E), dtype=tc.float32)          # (B, D, E)
    
#     p[..., 0::2] = tc.sin(kx[..., 0::2])               # (B, D, E)
#     p[..., 1::2] = tc.cos(kx[..., 1::2])               # (B, D, E)
#     return p







# class Image_DM_(nn.Module):
#     def __init__(self, C=3, H=128, W=128, c=256, E=8, Clamp=True):
#         super(Image_DM_, self).__init__()
#         self.C, self.H, self.W, self.c, self.E = C, H, W, c, E
#         self.f_s = 8
#         self.h, self.w = H // self.f_s, W // self.f_s
        
#         self.time_proj = nn.Sequential(
#             nn.Linear(self.c, self.c),
#             nn.GELU(),
#             nn.Linear(self.c, self.c)
#         )
        
#         self.Encoder = nn.Sequential(
#             nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.GELU(),
#             nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.GELU(),
#             nn.Conv2d(32, c, kernel_size=4, stride=2, padding=1),
#             nn.GroupNorm(8, c),
#             nn.GELU()
#         )
        
#         self.Decoder = nn.Sequential(
#             nn.ConvTranspose2d(c, 32, kernel_size=4, stride=2, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.GELU(),
#             nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.GELU(),
#             nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1)  # No activation here
#         )
        
#     def forward(self, x, t):
#         B, C, H, W = x.shape
        
#         # Create and project time embeddings
#         p_t = Sinusoidal_Embedding_1D_(t, E=self.c)  # (B, c)
#         p_t = self.time_proj(p_t).view(B, self.c, self.h, self.w)  # (B, c, h, w)
        
#         # Encoder with time embedding added to input
#         z_ = self.Encoder(x + p_t)  # (B, c, h, w)
        
#         # Decoder
#         x_ = self.Decoder(z_)  # (B, C, H, W)
        
#         return x_


# class SinusoidalTimeEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, t):
#         half_dim = self.dim // 2
#         embeddings = tc.exp(-tc.arange(half_dim, dtype=tc.float32) * (2 * tc.pi / half_dim))
#         embeddings = t[:, None] * embeddings[None, :]
#         return tc.cat([tc.sin(embeddings), tc.cos(embeddings)], dim=-1)
