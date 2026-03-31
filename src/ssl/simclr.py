import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

## loss definition 

class SupConLoss(nn.Module):
    def __init__(self,  
                 sim = 'cosine', 
                 tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 

    def forward(self, features, labels):
        B, _ = features.shape
        # calculate pair wise similarity 
        sim_mat = self.calculate_sim_matrix(features)
        # division by temperature
        sim_mat = F.log_softmax(sim_mat / self.tau, dim = -1) 
        sim_mat = sim_mat.clone().fill_diagonal_(torch.tensor(0.0))
        # calculating pair wise equal labels for pos pairs
        labels = labels.unsqueeze(1)
        labels_mask = (labels == labels.T).type(torch.float32)
        labels_mask.fill_diagonal_(torch.tensor(0.0))
        # calculating num of positive pairs for each sample
        num_pos = torch.sum(labels_mask, dim = -1)
        # masking out the negative pair log_softmax value
        pos_sim_mat = sim_mat * labels_mask 
        # summing log_softmax value over all positive pairs
        pos_pair_sum = torch.sum(pos_sim_mat, dim = -1)
        # averaging out the log_softmax value, epsilon = 1e-5 is to avoid division by zero
        pos_pairs_avg = torch.div(pos_pair_sum, num_pos + 1e-5)
        # final loss over all features in batch
        loss = -pos_pairs_avg.sum() / B
        return loss

    def calculate_sim_matrix(self, features):
        sim_mat = None
        if self.sim == "mse":
            sim_mat = -torch.cdist(features, features)
        elif self.sim == "cosine":
            features = F.normalize(features, dim = -1, p = 2)
            sim_mat = F.cosine_similarity(features[None, :, :], features[:, None, :], dim = -1)
        else: # bhattacharya coefficient
            features = F.normalize(features, dim = -1, p = 2)
            features = F.softmax(features, dim = -1)
            sqrt_feats = torch.sqrt(features) # sqrt of prob dist 
            sim_mat = sqrt_feats @ sqrt_feats.T
            
        # filling diagonal with -torch.inf as it will be cancel out while doing softmax
        sim_mat.fill_diagonal_(-torch.tensor(torch.inf))
        return sim_mat      
    
class SimCLR(nn.Module):
    def __init__(self, sim = 'cosine', tau = 1.0):
        super().__init__()
        self.sim = sim 
        self.tau = tau 
        self.supcon = SupConLoss(sim, tau)

    def forward(self, x, x_cap):
        B = x.shape[0]
        device = x.device 

        fake_label = torch.arange(0, B, device = device)
        fake_labels = torch.cat([fake_label, fake_label])

        x_full = torch.cat([x,x_cap], dim = 0)

        return self.supcon(x_full, fake_labels)
    
    def __repr__(self):
        return f"SimCLR(sim={self.sim}, tau={self.tau})"


class BYOL_mlp(nn.Module): # pred and proj net for carl
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.mlp(x)