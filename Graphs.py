import sys
import os

import numpy as np
from sklearn.decomposition import PCA

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from Interpreter import *

###############################################################################
# KD Tree

class KDTree:
    def __init__(self, points):
        """
        Initializes the KDTree.
        :param points: Tensor of shape (B, N, D) - batch of point clouds.
        """
        self.points = points
        self.batch_size, self.num_points, self.dim = points.shape
        self.tree = self.build_tree(points)
        
    def build_tree(self, points, depth=0):
        """
        Recursively builds the KD-tree.
        :param points: Tensor of shape (B, N, D)
        :param depth: Current tree depth (used for splitting dimension)
        :return: Sorted points forming a KD-tree
        """
        if points.shape[1] == 0:
            return None
        
        axis = depth % self.dim  # Select splitting dimension
        sorted_points, indices = tc.sort(points[:, :, axis], dim=1)  # Sort along the axis
        median_idx = sorted_points.shape[1] // 2  # Median index
        
        # Recursively build left and right subtrees
        left_tree = self.build_tree(points[:, :median_idx, :], depth + 1)
        right_tree = self.build_tree(points[:, median_idx + 1:, :], depth + 1)
        
        return {
            'point': points[:, median_idx, :],  # Root point
            'left': left_tree,
            'right': right_tree
        }
    
    def knn_search(self, query_points, k):
        """
        Performs k-nearest neighbors search.
        :param query_points: Tensor of shape (B, N, D) - query points
        :param k: Number of neighbors to find
        :return: (B, N, k) indices of k-nearest neighbors
        """
        batch_size, num_queries, _ = query_points.shape
        indices = tc.zeros((batch_size, num_queries, k), dtype=tc.long)

        for i in range(num_queries):  # Loop over each query point
            indices[:, i, :] = self._knn(self.tree, query_points[:, i, :], k, depth=0)
            
        return indices
    
    def _knn(self, tree, query, k, depth):
        """
        Recursively searches for k-nearest neighbors.
        :param tree: KD-tree structure
        :param query: Tensor of shape (B, D)
        :param k: Number of neighbors
        :param depth: Current depth in the KD-tree
        :return: Indices of k-nearest neighbors
        """
        if tree is None:
            return tc.full((query.shape[0], k), -1, dtype=tc.long)  # Return dummy indices
    
        axis = depth % self.dim
        go_left = query[:, axis] < tree['point'][:, axis]  # Shape (B,)
    
        # Initialize output storage
        best_indices = tc.zeros((query.shape[0], k), dtype=tc.long)
    
        for i in range(query.shape[0]):  # Iterate over batch dimension
            if go_left[i]:
                best_tree = tree['left']
            else:
                best_tree = tree['right']
            best_indices[i] = self._knn(best_tree, query[i].unsqueeze(0), k, depth + 1)
    
        return best_indices


    def adjacency_matrix(knn_indices, num_points):
        """
        Creates an adjacency matrix from KNN indices.
        :param knn_indices: Tensor of shape (B, N, k)
        :param num_points: Number of points N
        :return: Adjacency matrix (B, N, N)
        """
        B, N, k = knn_indices.shape
        A = tc.zeros((B, N, N), dtype=tc.float32)
    
        # Loop through each point and set adjacency matrix entries
        for i in range(k):
            A.scatter_(2, knn_indices[:, :, i].unsqueeze(2), 1)
    
        return A

###############################################################################
# Point Cloud Utilities

def FPS1_(X_, M, B_=None):
    """
    Fully vectorized Batched Farthest Point Sampling (FPS).
    
    Args:
        X_: (B, N, D) or (B*N, D) point cloud data
        M:  number of samples per batch
        B_: (B*N,) batch indices (optional, required for flattened input)
    
    Returns:
        sampled_pts: (B, M, D)
    """
    dev = X_.device

    # Flatten input if needed
    if B_ is None:
        B, N, D = X_.shape
        X_ = X_.view(-1, D)                                           # (B*N, D)
        BN = B*N
        B_ = tc.arange(B, device=dev).repeat_interleave(N)            # (B.N,)
        reshape = False
    else:
        BN, D = X_.shape
        B = B_.max().item() + 1
        reshape = True

    # Build a (B, N) mask of batch membership
    N_max = tc.bincount(B_).max().item()
    idx = tc.arange(BN, device=dev)                                 # (B.N,)
    batch_mask = tc.zeros((B, N_max), dtype=tc.bool, device=dev)    # (B, N_max) 
    X_b = tc.zeros((B, N_max, D), device=dev)                       # (B, N_max, D)
    
    for b in range(B):
        inds = (B_ == b).nonzero(as_tuple=True)[0]                  # (n,)
        n = inds.shape[0]
        batch_mask[b, :n] = True                                    # (n,)
        X_b[b, :n] = X_[inds]                                       # (n, D)

    N = batch_mask.sum(dim=1).max().item()                          # (N_max)

    # Initialize FPS
    B_idx = tc.arange(B, device=dev)                                # (B,)
    dist = tc.full((B, N), float('inf'), device=dev)                # (B, N)
    farthest_idx = tc.randint(0, N, (B,), device=dev)               # (B,)
    centroids = tc.zeros((B, M), dtype=tc.long, device=dev)         # (B, M)
    
    for i in range(M):
        centroids[:, i] = farthest_idx                              # (B,)
        centroids_xyz = X_b[B_idx, farthest_idx]                    # (B, D)
        # dists = tc.cdist(X_b, centroids)                          # (B, N)
        dists = ((X_b - centroids_xyz.unsqueeze(1))**2).sum(-1)     # (B, N)
        dist = tc.minimum(dist, dists)                              # (B, N)
        dist[~batch_mask] = -1  # Ignore padding
        farthest_idx = dist.max(dim=1)[1]                           # (B,)

    Y_ = X_b[B_idx.unsqueeze(1), centroids]                         # (B, M, D)
    
    if reshape:
        Y_ = Y_.view(B*M, D)                                        # (B*M, D)
    
    return Y_


def FPS_(X_, M, B_=None, Flat=False):
    """
    Fully vectorized Batched Farthest Point Sampling (FPS).

    Args:
        X_: (B*N, D) flattened point cloud data
        M:  int, number of samples per batch
        B_: (B*N,) batch indices

    Returns:
        Y_: (B*M, D) sampled points
    """
    dev = X_.device
    
    # Flatten input if needed
    if B_ is None:
        B, N, D = X_.shape
        X_ = X_.view(-1, D)                                         # (B*N, D)
        BN = B*N
        B_ = tc.arange(B, device=dev).repeat_interleave(N)          # (B.N,)
    else:
        BN, D = X_.shape
        B = B_.max().item() + 1
    
    # Compute sizes and batch offsets
    batch_sizes = tc.bincount(B_, minlength=B).to(dev)              # (B,)
    batch_offsets = tc.zeros(B, dtype=tc.long, device=dev)          # (B,)
    batch_offsets[1:] = batch_sizes.cumsum(dim=0)[:-1]              # (B,)
    
    # Init
    dist = tc.full((BN,), float("inf"), device=dev)                 # (B.N,)
    centroids = tc.zeros((B, M), dtype=tc.long, device=dev)         # (B, M)
    
    # Choose first point randomly per batch
    rand = (tc.rand(B, device=dev) * batch_sizes.float()).long()    # (B,)
    farthest = rand + batch_offsets                                 # (B,)
    
    for i in range(M):
        centroids[:, i] = farthest                                  # (B,)
        
        # Get current centroids for all points in batch
        c_ = X_[farthest[B_]]                                       # (B.N, D)
        d_ = ((X_ - c_)**2).sum(-1)                                 # (B.N,)
        dist = tc.minimum(dist, d_)                                 # (B.N,)
        
        # Compute per-batch argmax of dist using small loop
        farthest = tc.empty(B, dtype=tc.long, device=dev)           # (B,)
        for b in range(B):
            start = batch_offsets[b]                                # (B.N,)
            end = start + batch_sizes[b]                            # (B.N,)
            farthest[b] = start + dist[start:end].argmax()          # ()
            
    # Final sampled points
    flat_idx = centroids.view(-1)                                   # (B*M,)
    Y_ = X_[flat_idx]                                               # (B*M, D)
    
    if not Flat:
        Y_ = Y_.view(B, M, D)                                       # (B, M, D)
    
    return Y_


def KMeans_(X_, M, B_=None, iters=3, Flat=False):
    """
    Fully vectorized Batched K-Means for centroid extraction.

    Args:
        X_:     (B, N, D) or (B*N, D) input point cloud data
        M:      number of clusters (centroids) per batch
        B_:     (B*N,) batch indices (optional, for flattened input)
        iters:  number of K-means iterations

    Returns:
        Y_:     (B, M, D) cluster centroids per batch
    """
    dev = X_.device

    # Reshape if needed
    if B_ is None:
        B, N, D = X_.shape
        X_ = X_.reshape(-1, D)                                      # (B*N, D)
        B_ = tc.arange(B, device=dev).repeat_interleave(N)          # (B,)
    else:
        BN, D = X_.shape
        B = B_.max().item() + 1


    # Reconstruct padded batched data
    N_per_batch = tc.bincount(B_)
    N_max = N_per_batch.max().item()
    X_b = tc.zeros((B, N_max, D), device=dev)                       # (B, N_max, D)    
    mask = tc.zeros((B, N_max), dtype=tc.bool, device=dev)          # (B, N_max)

    for b in range(B):
        idx = (B_ == b).nonzero(as_tuple=True)[0]                   # (n,)
        n = idx.shape[0]
        X_b[b, :n] = X_[idx]                                        # (n, D)
        mask[b, :n] = 1                                             # (n,)

    # Random init of centroids: choose M random points per batch
    rand_idx = tc.randint(0, N_max, (B, M), device=dev)             # (B, M)
    centroids = X_b[tc.arange(B).unsqueeze(1), rand_idx]            # (B, M, D)
    
    for _ in range(iters):
        # Compute distances: (B, N, M)
        dists = tc.cdist(X_b, centroids)                            # (B, N, M)

        # Set large dist for padded entries
        dists[~mask] = float('inf')                                 # (B, N, M)

        # Cluster assignment
        assign = dists.argmin(dim=-1)                               # (B, N)

        # Compute new centroids
        centroids.fill_(0)                                          # (B, M, D)
        counts = tc.zeros(B, M, 1, device=dev)                      # (B, M, 1)
        
        # One-hot encode assignments: (B, N, M)
        OHE = F.one_hot(assign, num_classes=M).float()              # (B, N, M)
        
        # Apply mask: zero out padded points
        OHE *= mask.unsqueeze(-1)                                   # (B, N, M)
        
        # Compute cluster counts: (B, M, 1)
        counts = OHE.sum(dim=1, keepdim=True).transpose(1, 2)       # (B, M, 1)
        
        # Compute weighted sum of points per cluster
        # X_b: (B, N, D), assign_onehot: (B, N, M)
        centroids = tc.einsum('bnd,bnm->bmd', X_b, OHE)             # (B, M, D)
        
        # Normalize
        # centroids = centroids / counts.transpose(1, 2).clamp_min(1e-6)  # (B, M, D)
        centroids = centroids / (counts.clamp_min(1e-6))  # avoid divide by zero

    if Flat:
        centroids = centroids.view(B*M, D)                           # (B*M, D)
    
    return centroids


###############################################################################
# Graph Utilities

def Edg_Adj_(E_):
    """
    Convert a batched edge list to a batched adjacency matrix.
    
    Args:
            E_: Edge list tensor of shape (B, E, 2) where:
            B = batch size
            E = number of edges
            Last dimension contains [source, target] node indices
           
    Returns:
        A_: Adjacency matrix of shape (B, N, N) where N is the maximum node index + 1
    """
    # E_: (B, E, 2)
    B, E, _ = E_.shape
    
    # Find the maximum node index to determine N
    N = int(E_.max().item()) + 1
    
    # Initialize adjacency matrices
    A_ = tc.zeros((B, N, N), dtype=tc.bool, device=E_.device)
    
    # Create batch indices for advanced indexing
    Batch = tc.arange(B, device=E_.device).view(B, 1).expand(B, E)
    
    # Extract source and target indices
    From = E_[:, :, 0]  # (B, E)
    To =   E_[:, :, 1]  # (B, E)
    
    # Set edges in adjacency matrix using advanced indexing
    A_[Batch, From, To] = True
    
    # Make symmetric for undirected graph
    A_ = A_ | A_.transpose(1, 2)
    
    return A_


def Adj_Edg_(A_):
    # A_: (B, N, N)

    # A_[np.diag(A_)] = False     # Dont plot self connections

    # Get all valid edges
    i, j = np.where(A_)                         # (E,) x2
    
    # tc.stach

    # Stack the pairs of points to create edges (B, 2, D)    
    # L_ = np.stack([A_[i], A_[j]], axis=1)   # (B, 2, D)
    
    E_ = None
    return E_


def Batch_Graph_(X_, E_):
    """
    Vectorized processing of batched graphs with potentially missing nodes (NaNs).
    
    Args:
        X_: (B, N, D)    - Node features, possibly with NaNs
        E_: (B, E, 2)    - Local edge indices
    
    Returns:
        X_: (B.N, D)     - Valid node features
        E_: (2, B.E)     - Global edge indices
        B_: (B.N,)       - Batch index for each node

                """
    B, N, D = X_.shape
    _, E, _ = E_.shape

    # 1: Reshape X_
    X_ = X_.view(B*N, D)                                        # (B*N, D)
    
    # 2: Mask NaNs   
    mask = ~tc.isnan(X_).any(dim=-1)                            # (B*N)

    # 3: Filter out NaNs from X_
    X_ = X_[mask]                                               # (B.N, D)

    # 4: Batch vector to preserve batch index
    B_ = tc.arange(B).repeat_interleave(N)[mask]                # (B.N,)

    # 5: Build a mapping from (B, N) → global valid index
    I = tc.full((B*N,), -1, dtype=tc.long, device=X_.device)    # (B*N)
    I[mask] = tc.arange(mask.sum(), device=X_.device)           # (B*N)
    
    # 6: Flatten edges and add batch offset
    E_ += tc.arange(B)[:, None, None] * N                       # (B, 1, 1)
    E_ = E_.view(B*E, 2).T                                      # (2, B*E)
    
    # 7: Remap local indices to global indices using mapping I
    E_ = I[E_]                                                  # (2, B*E)
    
    # 8. Filter out edges that were connected to NaN points
    E_ = E_[:, (E_ != -1).all(dim=0)]                           # (2, B.E)

    return X_, E_, B_


def Plot_Graph_(E_, X_=None, B_=None, i=0):
    """
    Plots a single graph from a batch using edge list and node features.

    Args:
        E_: (B.E, 2) edge list (flattened)
        X_: (B.N, D) node features (optional)
        B_: (B.N,) node batch index (optional)
        i:  int, which graph in batch to visualize
    """
    E_ = E_.detach().cpu().numpy()
    if X_ is not None:
        X_ = X_.detach().cpu().numpy()
    if B_ is not None:
        B_ = B_.detach().cpu().numpy()

    # === Filter to batch i ===
    if B_ is not None:
        node_mask = (B_ == i)
        node_idx = np.nonzero(node_mask)[0]
        node_map = {old: new for new, old in enumerate(node_idx)}

        X_ = X_[node_idx] if X_ is not None else None

        # Filter edges: both ends must be in the same graph
        edge_mask = np.isin(E_[:, 0], node_idx) & np.isin(E_[:, 1], node_idx)
        E_ = E_[edge_mask]
        E_ = np.vectorize(node_map.get)(E_)  # reindex edges to local node indices

    N = X_.shape[0] if X_ is not None else E_.max() + 1

    # === Node Positions & Colors ===
    if X_ is not None:
        D = X_.shape[1]
        if D <= 3:
            C_ = np.zeros((N, 3))  # default color: black
        elif D == 6:
            C_ = X_[:, 3:]
        else:
            C_ = X_[:, 3:] if D > 3 else np.zeros((N, 3))
            C_ = Project_3D_(C_)
    else:
        D = 2
        t_ = np.arange(N) / N * 2 * np.pi
        X_ = np.stack([np.cos(t_), np.sin(t_)], axis=-1)
        C_ = np.zeros((N, 3))

    # === Edges ===
    lines = X_[E_][:, :, [0, 2 if D > 2 else 1]]
    edge_colors = (C_[E_[:, 0]] + C_[E_[:, 1]]) / 2
    lc = LineCollection(lines, colors=edge_colors, linewidths=1.0)

    # === Plot ===
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.scatter(X_[:, 0], X_[:, 2 if D > 2 else 1], c=C_, s=30, edgecolors='k', linewidths=0.5)

    ax.autoscale()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


    # for i in range(E):
    #     a, b = E_[i].T
    #     c_ = (C_[a] + C_[b])/2
    #     plt.plot(X_[[a, b], 0], X_[[a, b], 1], '-', color=tuple(c_))
    
    # for i in range(N):
    #     plt.plot(X_[i, 0], X_[i, 1], 'o', color=tuple(C_[i]))
    
    # plt.show()
    
    return

###############################################################################
# Contruct Graph

def KNN_(X_, k=5, d_=True, I_=True, Y_=False, A_=False, E_=False):
    # X: (B, N, D)
    B, N, D = X_.shape
    
    # Compute pairwise distances (using broadcasting)
    d_ = tc.cdist(X_, X_)                                           # (B, N, N)
        
    # Get indices of k nearest neighbors
    I_ = d_.topk(k, dim=-1, largest=False)[1]                       # (B, N, k)
    
    if Y_:
        # Use advanced indexing to gather the k nearest points
        batch_idx = tc.arange(B).view(-1, 1, 1).expand_as(I_)       # (B, N, k)
        Y_ = X_[batch_idx, I_]                                      # (B, N, k, D)
        
    if A_:
        A_ = tc.zeros((B, N, N), dtype=tc.bool)                     # (B, N, N)
        A_.scatter_(2, I_[:, :, 1:], True)
        
    if E_:
        I0 = tc.arange(N).view(1, N, 1).expand(B, N, k)             # (B, N, k)
        E_ = tc.stack([I0, I_], dim=-1).view(B, -1, 2)              # (B, E, 2)
    
    return d_, I_, Y_, I_, A_, E_


def DNN_(X_, d_max=None, k_max=10):
    # X: (B, N, D)
    B, N, D = X_.shape
    
    # Compute pairwise distances (using broadcasting)
    d_ = tc.cdist(X_, X_)                                   # (B, N, N)
    
    # if 
    d_max = d_.mean(axis=(1, 2), keepdims=True)             # (B,)
    
    A_ = d_ < d_max                                         # (B, N, N)
    
    E_ = tc.where(A_)                                       # (B.N.k.) x3
    
    # Get indices of k nearest neighbors
    # I = d.topk(k+1, dim=-1, largest=False)[1][:, :, 1:]     # (B, N, k)
    
    # Use advanced indexing to gather the k nearest points
    # batch_idx = tc.arange(B).view(-1, 1, 1).expand_as(I)    # (B, N, k)
    
    # Y = X[batch_idx, I]                                     # (B, N, k, D)
    
    # A = tc.zeros((B, N, N), dtype=tc.bool)                  # (B, N, N)
    
    # A.scatter_(2, I, True)
    
    # return Y, I, A
    return A_, E_


def RNN_(X_, k_avg=5, Adj=False):
    """
    Constructs a graph from a point cloud with approximately k_avg neighbors per node on average,
    but distributes connections according to point density.
    
    Args:
        X: Point cloud tensor of shape (B, N, D) where:
            B = batch size
            N = number of points
            D = dimensionality of each point
        k_avg: Target average number of neighbors per node
        
    Returns:
        A: Adjacency matrix of shape (B, N, N)
        E: Edge list as a tuple of indices (batch_indices, source_indices, target_indices)
        D: Pairwise distance matrix of shape (B, N, N)
    """
    dev = X_.device

    # Get dimensions
    B, N, D = X_.shape
    
    # Compute total number of allowed edges (k_avg * N per batch)
    E = k_avg * N
    
    # Compute pairwise distances
    d_ = tc.cdist(X_, X_)                                           # (B, N, N)
    
    # Create a mask for the upper triangular part (excluding diagonal)
    # T = N*(N-1)/2
    triu_indices = tc.triu_indices(N, N, offset=1).to(device=dev)   # (2, T)
    r_, c_ = triu_indices                                           # (T,)
    
    # Get the corresponding distances for each batch
    d_T = d_[:, r_, c_]                                             # (B, T)
    
    # # Get top-E indices for each batch
    _, I1 = tc.topk(d_T, E, dim=1, largest=False)                   # (B, E)

    # Use advanced indexing to gather the correct row and column indices
    r_ = r_[None].expand(B, -1)                                     # (B, T)
    c_ = c_[None].expand(B, -1)                                     # (B, T)
    
    # Now gather using advanced indexing with the top_k_indices
    # batch_dim x top_k_dim -> gather from triu indices
    From = tc.gather(r_, 1, I1)                                     # (B, E)
    To =   tc.gather(c_, 1, I1)                                     # (B, E)
    
    # Stack to create final edge tensor
    E_ = tc.stack([From, To], dim=2)                                # (B, E, 2)
    
    if Adj:
        A_ = Edg_Adj_(E_)                                           # (B, N, N)
        
    return E_


def Fully_Connected_(B, N, A_=False, E_=False):
    
    if A_:
        A_ = tc.ones((B, N, N))         # (B, N, N)
    
    if E_:
    
        # E_ = Adj_Edg_(A_)
    
        i_ = tc.arange(N)[None, :, None].expand(B, 1, N)        # (B, N, N)
        j_ = tc.arange(N)[None, None, :].expand(B, N, 1)        # (B, N, N)
        
        E_ = tc.stack([i_, j_], dim=-1)                         # (B, N, N, 2)
        E_ = E_.view(B, N**2, 2)                                # (B, N^2, 2)
        
    return A_, E_
    
# tc.manual_seed(0)

# B, N, D = 10, 100, 3
# X_ = tc.rand(B, N, D)#.view(-1, D)
# # B_ = tc.arange(B).repeat_interleave(N)            # (B.N,)
# B_ = None

# # Y_ = FPS_(X_, M=20, B_=B_)                        # (B, M, D)
# Y_ = KMeans_(X_, M=20, B_=B_, iters=3)                        # (B, M, D)

# for i in range(B):
#     plt.plot(X_[i, :, 0], X_[i, :, 1], 'o', markersize=10, color='black')
#     plt.plot(Y_[i, :, 0], Y_[i, :, 1], 'o', color='red')
#     plt.show()
