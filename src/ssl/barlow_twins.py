import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class BarlowTwinLoss(nn.Module):
    def __init__(self, lambd = 0.1):
        super().__init__()
        self.lambd = lambd

    def forward(self, za, zb): # za and zb are already batch normalized 
        N, D = za.shape

        C = torch.mm(za.T, zb) / N # DxD

        I = torch.eye(D, device=za.device)

        diff = (I - C).pow(2)

        diag_elem = torch.diag(diff)

        diff.fill_diagonal_(0.0)

        return diag_elem.sum() + self.lambd * diff.sum()

    def __repr__(self):
        return f"BT(lambda={self.lambd})"

class bt_proj(nn.Module):
    def __init__(self, in_features, barlow_hidden, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(in_features, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden),
                nn.ReLU(),
                nn.Linear(barlow_hidden, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden),
                nn.ReLU(),
                nn.Linear(barlow_hidden, proj_dim),
                nn.BatchNorm1d(proj_dim, affine=False)
            )

    def forward(self, x):  
        return self.proj(x)