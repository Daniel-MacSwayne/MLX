import sys
import os

import numpy as np

import torch as tc
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
# tc.autograd.set_detect_anomaly(True)

# import matplotlib.pyplot as plt

###############################################################################

class Token_Autoregressor_(nn.Module):
    """A class that gives a Model diffusion-based training methods."""
    
    def __init__(self, Model, D:tuple):
        super(Token_Autoregressor_, self).__init__()
        
        self.D = D
        self.Model = Model
        
        return
   
    
    def forward(self, x:tc.tensor):
        # x (B, D...)
    
        p = self.Model(x)     # (B, D...)
        
        return p

    
    def Generate_(self, x:tc.tensor=None):
        """Sample the data space and move it towards the data manifold"""
        # x: (B, D...)
        
        return x
    
    
    def Train_(self, Dataloader, num_epochs:int=1, lr:float=1e-4):
        
        Loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(self.Model.parameters(), lr=lr)
        # B = Dataloader.batch_size
        
        for i in range(num_epochs):
            for j, x in enumerate(Dataloader):
                    
                B, D = x.shape[0], x.shape[1:]
                
                # x_t, ϵ = self.Forward_(x_0, t)                      # (B, D...)
                # x_t_1, ϵ_θ = self.Backward_(x_t, t)                 # (B, D...)

                L = Loss_fn(ϵ_θ, ϵ)                                 # (1,)
                
                L.backward()
                opt.step()
                opt.zero_grad()
                
                if (j+1) % 1 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(i+1, num_epochs, (j+1), len(Dataloader), L.item()))
                    
                    # plt.imshow(x_0.numpy()[0, 0]), plt.show()
                    # plt.imshow(x_t.numpy()[0, 0]), plt.show()
                    # plt.imshow(x_t_1.detach().numpy()[0, 0]), plt.show()
                    
                    # plt.imshow(ϵ.numpy()[0, 0]), plt.show()
                    # plt.imshow(ϵ_θ.detach().numpy()[0, 0]), plt.show()
                    
                    # tc.save(self.state_dict(), "Model.pth")
        return
    

class Diffusion_Autoregressor_(nn.Module):
    """A class that gives a Model diffusion-based training methods."""
    
    def __init__(self, DM):
        super(Diffusion_Autoregressor_, self).__init__()
        
        self.DM = DM
        self.D = self.DM.D
        
        return
   
    
    # def forward(self, x:tc.tensor):
    #     # x (B, D...)
    
    #     p = self.Model(x)     # (B, D...)
        
    #     return p

    
    def Generate_(self, x:tc.tensor=None, L=10, s_0=None, Use_Noise=True):
        """Autoregressively generate samples conditioned on previous samples."""
        # x: (B, D...)
        
        # B, D = x.shape[0], x.shape[1:]
        
        S_ = tc.zeros(((L,) + self.DM.D))             # (L, D...)
        # S_[0] = self.DM.Sample_(x_T=None, c=None)
        S_[0, 0] = 1
        # S_[0] = tc.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # S_[1] = tc.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.])
        # S_[0] = s_0
        
        for i in range(1, L):
            c = S_[i-1][None]         # (1, D...)
            
        # for i in range(2, L):
            # c = S_[i-2:i].reshape((1, -1))         # (1, 2D)

            S_[i] = self.DM.Sample_(x_T=None, c=c, Use_Noise=Use_Noise)[0]
            
        return S_






    