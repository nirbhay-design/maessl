import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class VICRegLoss(nn.Module):
    def __init__(self, _lambda = 25.0, mu = 25.0, nu = 1.0):
        super().__init__()
        self._lambda = _lambda
        self.mu = mu    
        self.nu = nu

    def off_diagonal(self, x):
        n, _ = x.shape 
        return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

    def forward(self, za, zb):
        sim_loss = F.mse_loss(za, zb)

        za_var = torch.sqrt(za.var(dim = 0) + 1e-04)
        zb_var = torch.sqrt(zb.var(dim = 0) + 1e-04)
        var_loss = 0.5 * (torch.mean(F.relu(1-za_var)) + torch.mean(F.relu(1-zb_var)))

        za = za - za.mean(dim = 0)
        zb = zb - zb.mean(dim = 0)

        N, D = za.shape 
        cov_za = (za.T @ za) / (N - 1)
        cov_zb = (zb.T @ zb) / (N - 1)

        cov_loss = self.off_diagonal(cov_za).pow(2).sum().div(D) +\
                    self.off_diagonal(cov_zb).pow(2).sum().div(D)

        return self._lambda * sim_loss + self.mu * var_loss + self.nu * cov_loss 

    def __repr__(self):
        return f"VICReg(lambda={self._lambda}, mu={self.mu}, nu={self.nu})" 

class vicreg_proj(nn.Module):
    def __init__(self, in_features, barlow_hidden, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(in_features, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden),
                nn.ReLU(),
                nn.Linear(barlow_hidden, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden),
                nn.ReLU(),
                nn.Linear(barlow_hidden, proj_dim)
            )

    def forward(self, x):  
        return self.proj(x)

