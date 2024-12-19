import sys
import os

import numpy as np


import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt

# import Architectures

###############################################################################

class Autoencoder_(nn.Module):
    def __init__(self, Model):
        super(Autoencoder_, self).__init__()
        
        self.Model = Model
        
        return
    
    def Train_(self, Dataset, I=100):
        
        Loss_fn = nn.MSELoss()
        opt = optim
        
        for i in range(I):
            
            x = 0
            x_, z_ = self.Model(x)
            
            L = Loss_fn(x_, x)
            
            L.backward()
            opt.step()
            opt.zero_grad()            
            
            
        return
        













