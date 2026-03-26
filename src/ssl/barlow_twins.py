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

def train_bt_sc(
        model, energy_model, train_loader, lossfunction, 
        optimizer, energy_optimizer, opt_lr_schedular, 
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    print(f"### Barlow Twins with SC-net Training begins")

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

            feat, proj_feat = output["features"], output["proj_features"]
            feat_cap, proj_feat_cap = output_cap["features"], output_cap["proj_features"]

            esample = energy_model.langevin_sampling(feat)
            esample_cap = energy_model.langevin_sampling(feat_cap)

            loss_con = lossfunction(proj_feat, proj_feat_cap) + F.mse_loss(feat, esample.detach()) + F.mse_loss(feat_cap, esample_cap.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

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

    # print("### TSNE starts")
    # make_tsne_for_dataset(model, test_loader, device_id, 'bt-sc', return_logs = return_logs, tsne_name = tsne_name)

    # print("### MLP training begins")

    # train_mlp(
    #     model, mlp, train_loader_mlp, test_loader, 
    #     lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
    #     device_id, eval_id, return_logs = return_logs)

    return model

def train_bt(
        model, train_loader, lossfunction, 
        optimizer, opt_lr_schedular, 
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    print(f"### Barlow Twins Training begins")

    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
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