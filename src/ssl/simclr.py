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

        self.supcon = SupConLoss(sim, tau)

    def forward(self, x, x_cap):
        B = x.shape[0]
        device = x.device 

        fake_label = torch.arange(0, B, device = device)
        fake_labels = torch.cat([fake_label, fake_label])

        x_full = torch.cat([x,x_cap], dim = 0)

        return self.supcon(x_full, fake_labels)

def train_scalre( # Score Alignment for Representation Learning
        model, energy_model, train_loader,
        lossfunction, optimizer, energy_optimizer, opt_lr_schedular, 
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    

    print(f"### ScAlRe Training begins")
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    energy_model = energy_model.to(device)
    for epochs in range(n_epochs):
        model.train()
        energy_model.train()
        cur_loss = 0
        en_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            output = model(data)
            output_cap = model(data_cap)

            proj_feat = output["proj_features"]
            proj_feat_cap = output_cap["proj_features"]

            feat = output["features"]
            feat_cap = output_cap["features"] 

            esample = energy_model.langevin_sampling(feat)
            esample_cap = energy_model.langevin_sampling(feat_cap)
            
            loss_con = lossfunction(proj_feat, proj_feat_cap) + F.mse_loss(feat, esample.detach()) + F.mse_loss(feat_cap, esample_cap.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            # training energy model
            # pos_energy = energy_model(feat.detach(), feat_cap.detach())
            # neg_energy = energy_model(esample.detach(), esample_cap.detach())

            energy_loss = 0.5 * (energy_model.dsm_loss(feat.detach()) + energy_model.dsm_loss(feat_cap.detach()))

            energy_optimizer.zero_grad()
            energy_loss.backward()
            energy_optimizer.step()
            
            cur_loss += loss_con.item() / (len_train)
            en_loss += energy_loss.item() / len_train

            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), en_loss = energy_loss.item(), GPU = device_id)

        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f} energy_loss: {en_loss:.3f}")

    return model

def train_simclr(
        model, train_loader, lossfunction, 
        optimizer, opt_lr_schedular, 
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    

    print(f"### SimCLR Training begins")
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            
            output = model(data)
            output_cap = model(data_cap)

            feats, proj_feat = output["features"], output["proj_features"]
            feats_cap, proj_feat_cap = output_cap["features"], output_cap["proj_features"]

            loss_con = lossfunction(proj_feat, proj_feat_cap)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model

