import sys
import os

import numpy as np


import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt

# from Architectures import *
# from Dataloaders import *
# from Trainer import *
from Loss_Functions import *

###############################################################################

class Autoencoder_(nn.Module):
    """A class that gives a Model Autoencoder-based training methods."""

    def __init__(self, Model, Loss_='MSE', Plot_=False):
        super(Autoencoder_, self).__init__()
        
        if Loss_ == 'MSE':
            def Loss_(x, x_, z_):
                L = nn.MSELoss()(x, x_)
                # L = L_(x, x_)
                return L
            
        elif Loss_ == 'VAE':
            self.Loss_ = VAE_Loss_()
        
        self.Model = Model
        self.Loss_ = Loss_
        self.Plot_ = Plot_
        
        return
    
    
    def forward(self, x:tc.tensor, c=None):
        # x:    (B, D...)
        # c:    [...]
        
        if c is None or tc.isnan(c).all():
            x_, z_ = self.Model(x)              # (B, D...), (B, d...)
        else:
            x_, z_ = self.Model(x, c)           # (B, D...), (B, d...)

        return x_, z_
    
    
    def Train_(self, Dataloader, num_epochs:int=1, lr:float=1e-4):
        
        opt = optim.Adam(self.Model.parameters(), lr=lr)   
        
        for i in range(num_epochs):
            for j, (x, y, c) in enumerate(Dataloader):
                # x: (B, D1...)
                # y: (B, D2...)
                # c: (B, C...)
                
                B, D = x.shape[0], x.shape[1:]
                
                x_, z_ = self(x, c)             # (B, D...), (B, d...)
                
                L = self.Loss_(x, x_, z_)       # (1,)
                
                L.backward()
                opt.step()
                opt.zero_grad()
                
                if (j+1) % 1 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(i+1, num_epochs, j+1, len(Dataloader), L.item()))
                    
                    if self.Plot_ != False:
                        self.Plot_(x, x_, z_)
        return
        

    def Train_Encoder_(self, Dataloader, num_epochs:int=1, lr:float=1e-4):
        
        Loss_fn = nn.L1Loss()
        opt = optim.Adam(self.Model.parameters(), lr=lr)
        
        for i in range(num_epochs):
            for j, (x, c) in enumerate(Dataloader):
               # x: (B, D...)
               # c: (B, C...)
                
                B, D = x.shape[0], x.shape[1:]
                
                z = tc.eye(10).repeat(10, 1)    # (B, 10)
                
                z_ = self.Model.Encoder(x, c)   # (B, d...)

                L = Loss_fn(z_, z)              # (1,)
                
                L.backward()
                opt.step()
                opt.zero_grad()
                
                if (j+1) % 1 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(i+1, num_epochs, j+1, len(Dataloader), L.item()))
                    
                    # tc.save(AE.state_dict(), "Digit_AE.pth")
        return


    def Train_Decoder_(self, Dataloader, num_epochs:int=1, lr:float=1e-4):
        
        Loss_fn = nn.L1Loss()
        opt = optim.Adam(self.Model.parameters(), lr=lr)
        
        for i in range(num_epochs):
            for j, x in enumerate(Dataloader):
               # x: (B, D...)
                
                B, D = x.shape[0], x.shape[1:]
                
                z_ = tc.eye(10).repeat(10, 1)    # (B, 10)
                
                x_ = self.Model.Decoder(z_)      # (B, d...)

                # for k in range(100):
                #     plt.imshow(x[k, 0].numpy()), plt.show()
                    
                # print(z_)
                # sys.exit()

                L = Loss_fn(x_, x)              # (1,)
                
                L.backward()
                opt.step()
                opt.zero_grad()
                
                if (j+1) % 1 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(i+1, num_epochs, j+1, len(Dataloader), L.item()))
                    plt.imshow(x_[4].permute(1, 2, 0).detach().numpy()), plt.show()

                    
                    # tc.save(AE.state_dict(), "Digit_AE.pth")
        return












