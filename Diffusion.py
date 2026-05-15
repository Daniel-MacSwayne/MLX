import sys
import os

import numpy as np

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
# tc.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

###############################################################################

class Diffusion_(nn.Module):
    """A class that gives a Model diffusion-based training methods."""
    
    def __init__(self, Model, D:tuple, T:int=250, Noise_Type='Cosine', β_min=0.0001, β_max=0.02, Plot_=False):
        super(Diffusion_, self).__init__()
        
        self.D = D          # Data Dimensionality. Arbitrary Tuple Shape (D...)
        self.T = T          # Timesteps. int
        self.Model = Model
        self.Noise_Schedule_(β_min, β_max, Type=Noise_Type)
        
        self.Plot_ = Plot_
        
        return
   
    
    def Noise_Schedule_(self, β_min=0.0001, β_max=0.02, s=0.008, Type:str='Linear'):

        self.β_min = β_min
        self.β_max = β_max
        
        if Type == 'Linear':
            self.β_ = tc.linspace(self.β_min, self.β_max, self.T)       # (T,)
            self.α_ = 1 - self.β_                                       # (T,)
            self.αc_ = tc.cumprod(self.α_, axis=0)                      # (T,)
            
        elif Type == 'Quadratic':
            t_ = tc.linspace(0, 1, self.T+1)                            # (T+1)
            αc_ = 1 - t_ ** 2                                           # (T+1,)
            self.α_ =  αc_[1:] / αc_[:-1]                               # (T,)
            self.β_ = 1 - self.α_                                       # (T,)
            self.β_ = tc.clamp(self.β_, β_min, β_max)                   # (T,) 
            self.αc_ = αc_[1:]                                          # (T,)       
            
        elif Type == 'Cosine':
            t_ = tc.linspace(0, 1, self.T+1)                            # (T+1)
            αc_ = tc.cos((t_ + s)/(1 + s) * tc.pi/2) ** 2               # (T+1,)
            αc_ = αc_ / αc_[0]                                          # (T+1,)
            self.α_ =  αc_[1:] / αc_[:-1]                               # (T,)
            self.β_ = 1 - self.α_                                       # (T,)
            self.αc_ = αc_[1:]                                          # (T,)
            
        else:
            print('No Noise Schedule Set')
            
        return

      
    def forward(self, x_t:tc.tensor, t:tc.tensor, c=None):
        # x:    (B, D...)
        # t:    (B, 1)
        # c:    [...]
        assert x_t.shape[0] == t.shape[0], f"{x_t.shape}, {t.shape}"
        assert x_t.shape[1:] == self.D,    f"{x_t.shape}, {self.D}"
    
        if c is None:
            ϵ_θ = self.Model(x_t, t)        # (B, D...) Predicted Absolute Noise
        else:
            ϵ_θ = self.Model(x_t, t, c)     # (B, D...) Predicted Absolute Noise        
        
        assert x_t.shape == ϵ_θ.shape, f"{x_t.shape}, {ϵ_θ.shape}"

        return ϵ_θ

    
    def Forward_(self, x_0:tc.tensor, t:tc.tensor):
        """Forward Diffusion Process"""
        # x_0:  (B, D...)
        # t:    (B)
        assert x_0.shape[0] == t.shape[0], f"{x_0.shape}, {t.shape}"
        assert x_0.shape[1:] == self.D,    f"{x_0.shape}, {self.D}"

        B = t.shape[0]
        # B, D = t.shape[0], x_0.shape[1:]
        
        t = t.reshape((B,)+(1,)*len(self.D))                            # (B, 1...)
        
        ϵ = tc.normal(0, 1, (B,)+self.D)                                # (B, D...) Standard Gaussian Noise
        
        x_t = self.αc_[t]**0.5 * x_0 + (1 - self.αc_[t])**0.5 * ϵ       # (B, D...)

        assert x_0.shape == x_t.shape, f"{x_0.shape}, {x_t.shape}"

        return x_t, ϵ
    
    
    def Backward_(self, x_t:tc.tensor, ϵ_θ:tc.tensor, t:tc.tensor, c=None, Method='DDIM'):
        """Reverse Diffusion Process"""
        # x_t:  (B, D...)
        # t:    (B,)
        # c:    [...]
        assert x_t.shape[0] == t.shape[0],  f"{x_t.shape}, {t.shape}"
        assert x_t.shape[1:] == self.D,     f"{x_t.shape}, {self.D}"
        assert x_t.shape == ϵ_θ.shape,      f"{x_t.shape}, {ϵ_θ.shape}"

        B = t.shape[0]
        
        if Method == 'DDPM':
            ϵ = tc.normal(0, 1, (B,)+self.D)                                # (B, D...) Standard Gaussian Noise
            # ϵ_θ = self(x_t, t, c)                                         # (B, D...)
    
            m = (t != 1) #* Use_Noise                                       # (B,)          
            t = t.reshape((B,)+(1,)*len(self.D))                            # (B, 1...)
            
            # x_t-1
            x_t_1 = x_t                                                     # (B, D...)
            x_t_1 -= self.β_[t]*(1 - self.αc_[t])**-0.5 * ϵ_θ               # (B, D...)
            x_t_1 *= self.α_[t]**-0.5                                       # (B, D...)
            x_t_1[m] += self.β_[t][m]**0.5 * ϵ[m]                           # (B, D...)

            
        elif Method == 'DDIM':
            
            # x_t-1
            x_t_1 = x_t                                                     # (B, D...)
            x_t_1 += ϵ_θ * (self.αc_[t]/self.αc_[t-1] - self.αc_[t])**0.5   # (B, D...)
            x_t_1 -= ϵ_θ * (1 - self.αc_[t])**0.5                           # (B, D...)
            x_t_1 *= (self.αc_[t-1]/self.αc_[t])**0.5                       # (B, D...)
            
            x_0_ = x_t - (1 - self.αc_[t])**0.5 * ϵ_θ                       # (B, D...)
            x_0_ *= self.αc_[t]**-0.5                                       # (B, D...)
            
        
        x_0_ = x_t - (1 - self.αc_[t])**0.5 * ϵ_θ                           # (B, D...)
        x_0_ *= self.αc_[t]**-0.5                                           # (B, D...)
        
        assert x_t.shape == x_0_.shape,     f"{x_t.shape}, {x_0_.shape}"
        assert x_t.shape == x_t_1.shape,    f"{x_t.shape}, {x_t_1.shape}"
    
        return x_t_1, x_0_, #ϵ_θ
    
    
    def Sample_(self, x_T:tc.tensor=None, c=None, Method='DDIM'):
        """Sample the data space and move it towards the data manifold"""
        # x_T:  (B, D...)
        # c:    (B, C...)

        if x_T == None:
            x_T = tc.normal(0, 1, (1,)+self.D)                              # (1, D...) Standard Gaussian

        assert x_T.shape[1:] == self.D,     f"{x_T.shape}, {self.D}"
        B = x_T.shape[0]
        
        X_ = tc.zeros((self.T, B,)+self.D)                                  # (T, B, D...)
        X_[-1] = x_T                                                        # (B, D...)
        
        t_ = tc.arange(self.T-1, 0, -1)[:, None]                            # (T, 1)
        
        for t in t_:

            print(t_.shape, t.shape, X_[t[0]].shape)#, ϵ_θ.shape)
            
            ϵ_θ = self(X_[t[0]], t, c)                                      # (B, D...)
            X_[t[0]-1] = self.Backward_(X_[t[0]], ϵ_θ, t, c, Method)[0]     # (B, D...)


        
        X_ = X_.transpose(0, 1)                                             # (B, T, D...)

        return X_
    
    
    def Train_(self, Dataloader, num_epochs:int=1, lr:float=1e-4):
        
        Loss_fn = nn.MSELoss()
        opt = optim.Adam(self.Model.parameters(), lr=lr)
        # B = Dataloader.batch_size
        
        for i in range(num_epochs):
            for j, (x_0, y, c) in enumerate(Dataloader):
                # x: (B, D1...)
                # y: None
                # c: (B, C...)
                assert x_0.shape[1:] == self.D,    f"{x_0.shape}, {self.D}"
                B = x_0.shape[0]
                # B, D = x_0.shape[0], x_0.shape[1:]
                
                t = tc.randint(self.T - 1, (B,))                            # (B)
                
                x_t, ϵ = self.Forward_(x_0, t)                              # (B, D...)
                ϵ_θ = self(x_t, t, c)                                       # (B, D...)
                # x_t_1, ϵ_θ, x_0_ = self.Backward_(x_t, t, c, Use_Noise, One_Shot) # (B, D...)

                L = Loss_fn(ϵ_θ, ϵ)                                         # (1,)
                
                L.backward()
                opt.step()
                opt.zero_grad()
                
                if (j+1) % 1 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(i+1, num_epochs, (j+1), len(Dataloader), L.item()))
                    
                    # if self.Plot_:
                    #     self.Plot_(x_0, x_0_, x_t, x_t_1, ϵ, ϵ_θ, c)
                    
                    # plt.imshow(x_0.numpy()[0, 0]), plt.show()
                    # plt.imshow(x_t.numpy()[0, 0]), plt.show()
                    # plt.imshow(x_t_1.detach().numpy()[0, 0]), plt.show()
                    
                    # plt.imshow(ϵ.numpy()[0, 0]), plt.show()
                    # plt.imshow(ϵ_θ.detach().numpy()[0, 0]), plt.show()
                    
                    # tc.save(self.state_dict(), "Model.pth")
        return






