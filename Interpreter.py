import sys
import os

import numpy as np
import scipy
from sklearn.decomposition import PCA

import torch as tc
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from tqdm import tqdm
# tc.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection

###############################################################################

def Project_3D_(X, projection_mode='pca', output_mode='sigmoid', independent_batches=True):
    # X: (B, N..., D)

    if not independent_batches:
        X = X[None]                                         # (1, B...N..., D)

    shape = X.shape
    B = shape[0]
    N = shape[1:-1]
    D = shape[-1]
    
    if isinstance(X, tc.Tensor):
        X = X.detach().cpu().numpy()                        # (B, N..., D)
        
    if projection_mode == 'pca':
        C_ = np.zeros((B,) + N + (3,))                      # (B, N..., 3)
        var = np.zeros((B, 3))                      # (B, 3)

        for i in range(B):
            x = X[i].reshape(-1, D)                         # (N, D)
            
            pca = PCA(n_components=3)
            c = pca.fit_transform(x)                        # (N, 3)

            c = c.reshape(N + (3,))                         # (N..., 3)
            c = c[..., [1, 0, 2]]                           # (N..., 3)
            C_[i] = c                                       # (N..., 3)
            var[i] = pca.explained_variance_ratio_ # (3,)
    
    elif projection_mode == 'mean':
        # w = np.ones((D, 3)) * 3/D                           # (D, 3)
        
        w = np.zeros((D, 3))                                # (d, 3)
        w[:D//3, 0] = 1.0    # First third of dims → Red
        w[D//3:2*D//3, 1] = 1.0  # Middle third → Green
        w[2*D//3:, 2] = 1.0  # Last third → Blue
        
        C_ = np.einsum('...d,dc->...c', X, w)               # (B, N..., 3)
        
    elif projection_mode == 'random':
        w = np.random.rand(D, 3).reshape(D, 3)              # (D, 3)
        w = w / np.linalg.norm(w, axis=0)                   # (D, 3)
        C_ = np.einsum('...d,dc->...c', X, w)               # (B, N..., 3)

    
    if not independent_batches:
        C_ = C_[0]                                          # (B, N..., 3)

    
    if output_mode == 'clip':
        C_ = np.clip(C_, 0, 1)
        
    elif output_mode == 'sigmoid':
        C_ = scipy.special.expit(C_)
        
    if projection_mode == 'pca':
        return C_, var
    else:
        return C_