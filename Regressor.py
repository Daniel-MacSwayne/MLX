import sys
import os

import numpy as np


import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt

###############################################################################

class Regressor_(nn.Module):
    """A class that gives a Model Autoencoder-based training methods."""

    def __init__(self, Model, Loss_='L2', Plot_=False):
        super(Regressor_, self).__init__()
        
        self.Model = Model
        self.Loss_ = Loss_
        self.Plot_ = Plot_
        
        if Loss_ == 'L2':
            def Loss_(y_, y, x, c):
                L = F.mse_loss(y_, y)
                return L
            
        self.Loss_ = Loss_

        try:
            self.Reg_ = self.Model.Reg_
        except:
            self.Reg_ = False
            
            
        
        return
    
    
    def forward(self, x:tc.tensor, c=None):
        # x:    (B, D1...)
        # c:    [B, C...]
        
        if tc.isnan(c).all():
        # if c is None:
            y_ = self.Model(x)                  # (B, D2...)
        else:
            y_ = self.Model(x, c)               # (B, D2...)

        return y_
    
    
    def Train_(self, Dataloader, 
               num_epochs:int=1, 
               lr:float=1e-4, 
               optimizer=tc.optim.Adam, 
               scheduler=tc.optim.lr_scheduler.StepLR,
               gamma=1
               ):
        """
        Args:
            Dataloader: torch DataLoader
            num_epochs: number of training epochs
            lr: default learning rate
            optimizer_cls: torch.optim class (e.g. torch.optim.SGD, Adam, etc.)
                           If None, defaults to Adam.
            scheduler_cls: torch.optim.lr_scheduler class (e.g. StepLR, CosineAnnealingLR, etc.)
                           If None, no scheduler is used.
            scheduler_kwargs: extra kwargs for scheduler
        """
        # Install optimizer
        opt = optimizer(self.Model.parameters(), lr=lr)
    
        # Optional scheduler
        sched = scheduler(opt, step_size=1, gamma=gamma)
    
        # x, c, y = Dataloader.dataset[0]
    
        for i in range(num_epochs):
            for j, (x, y, c) in enumerate(Dataloader):
                # x: (B, D1...)
                # y: (B, D2...)
                # c: (B, C...)
                # B, D = x.shape[0], x.shape[1:]
                if i == 0 and j == 0:
                    print(x.shape, y.shape)
    
                # Forward Pass
                y_ = self(x, c)                         # (B, D2...)
                
                # Loss
                L = self.Loss_(y_, y, x, c)             # (1,)
    
                # Regularizer
                if self.Reg_:
                    L += self.Reg_()                    # (1,)
                    
                # Backward Pass
                L.backward()
                opt.step()
                opt.zero_grad()
    
                if (j+1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, LR: {:.4f}'.format(
                        i+1, num_epochs, 
                        j+1, len(Dataloader), 
                        L.item(),
                        sched.get_last_lr()[0]))
    
                    if self.Plot_ is not False:
                        self.Plot_(x, y, y_, c)
    
                # Scheduler step (typically per epoch)
                if sched is not None:
                    sched.step()
                
        return










