import sys
import os

import numpy as np

import torch as tc
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# import matplotlib.pyplot as plt

###############################################################################

class Diffusion_(nn.Module):
    def __init__(self, Model, D:int, T:int=250, Noise_Type='Cosine', β_min=0.0001, β_max=0.02):
        super(Diffusion_, self).__init__()
        
        self.D = D
        self.T = T
        self.Model = Model
        self.Noise_Schedule_(β_min, β_max, Type=Noise_Type)
        
        return
    
    def Noise_Schedule_(self, β_min=0.0001, β_max=0.02, s=0.008, Type:str='Linear'):

        self.β_min = β_min
        self.β_max = β_max
        
        if Type == 'Linear':
            self.β_ = tc.linspace(self.β_min, self.β_max, self.T)   # (T,)
            self.α_ = 1 - self.β_                                   # (T,)
            self.αc_ = tc.cumprod(self.α_, axis=0)                  # (T,)
            
        elif Type == 'Quadratic':
            t_ = tc.linspace(0, 1, self.T+1)                        # (T+1)
            αc_ = 1 - t_ ** 2                                       # (T+1,)
            self.α_ =  αc_[1:] / αc_[:-1]                           # (T,)
            self.β_ = 1 - self.α_                                   # (T,)
            self.β_ = tc.clamp(self.β_, β_min, β_max)               # (T,) 
            self.αc_ = αc_[1:]                                      # (T,)       
            
        elif Type == 'Cosine':
            t_ = tc.linspace(0, 1, self.T+1)                        # (T+1)
            αc_ = tc.cos((t_ + s)/(1 + s) * tc.pi/2) ** 2           # (T+1,)
            αc_ = αc_ / αc_[0]                                      # (T+1,)
            self.α_ =  αc_[1:] / αc_[:-1]                           # (T,)
            self.β_ = 1 - self.α_                                   # (T,)
            self.αc_ = αc_[1:]                                      # (T,)
            
        else:
            print('No Noise Schedule Set')
            
        return

    
    def forward(self, x:tc.tensor, t:tc.tensor):
        # x (B, D)
        # t (B, 1)
    
        ϵ_θ = self.Model(x, t)     # (B, D) Predicted Absolute Noise
        
        return ϵ_θ

    
    def Forward_(self, x_0:tc.tensor, t:tc.tensor):
        # x_0 (B, D)
        # t (B, 1)
        
        B = t.shape[0]
        D = x_0.shape[1]
        
        ϵ = tc.normal(0, 1, (B, D))      # (B, D) Standard Gaussian Noise
        
        x_t = self.αc_[t]**0.5 * x_0 + (1 - self.αc_[t])**0.5 * ϵ      # (B, D)

        return x_t, ϵ
    
    
    def Backward_(self, x_t:tc.tensor, t:tc.tensor):
        # x_t (B, D)
        # t (B, 1)
        
        B = t.shape[0]
        D = x_t.shape[1]
        
        ϵ = tc.normal(0, 1, (B, D))                 # (B, D) Standard Gaussian Noise
        ϵ_θ = self(x_t, t)                          # (B, D)
        
        # x_t-1
        x_t_1 = x_t - self.β_[t]*(1 - self.αc_[t])**-0.5 * ϵ_θ     # (B, D)
        x_t_1 *= self.α_[t]**-0.5
        x_t_1 += self.β_[t]**0.5 * ϵ

        return x_t_1, ϵ_θ
    
    
    def Sample_(self, x_T:tc.tensor=None):
        
        if x_T == None:
            x_T = tc.normal(0, 1, (1, self.D))       # (1, D) Standard Gaussian
            
        # x_t = X_T
        x_ = tc.zeros((self.T, self.D))
        x_[-1] = x_T
        
        # t_ = tc.arange(0, self.T)
            
        for t in range(self.T, 1):
            ϵ_θ = self(x_[t:t+1], t)                          # (B, D)
            x_[t-1] = self.α_[t]**-0.5 * (x_[t] - self.β_**0.5 * ϵ_θ)
            
        return x_
    
    
    def Get_Data_Batch_(self, S_, B:int=20):
        
        N = S_.shape[0]
        x = S_[tc.randint(N, (B,))]  # (B, D)
        
        return x
    

    def Train_(self, S_, B:int=20, I:int=100, LR=0.001):
        
        Loss = nn.MSELoss()
        opt = optim.Adam(self.Model.parameters(), lr=LR)   
        
        for i in range(I):
            print(i)
            # progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            
            x_0 = self.Get_Data_Batch_(S_, B)               # (B, D)
            t = tc.randint(self.T - 1, (B, 1))     # (B, 1)
            
            x_t, ϵ = self.Forward_(x_0, t)              # (B, D)
            x_t_1, ϵ_θ = self.Backward_(x_t, t)         # (B, D)
            
            L = Loss(ϵ_θ, ϵ)
            
            L.backward()
            opt.step()
            opt.zero_grad()
        
        return
        
    









