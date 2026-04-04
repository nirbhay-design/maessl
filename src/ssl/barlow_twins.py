import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributed.nn.functional as dist_F
import torch.distributed as dist 

class BarlowTwinLoss(nn.Module):
    def __init__(self, lambd = 0.1):
        super().__init__()
        self.lambd = lambd

    def forward(self, za, zb): # za and zb are already batch normalized 
        if dist.is_initialized():
            za = torch.cat(dist_F.all_gather(za), dim = 0)
            zb = torch.cat(dist_F.all_gather(zb), dim = 0)

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
    
def train_bt(
        model, train_loader, loss_base, 
        optimizer, opt_lr_schedular, scaler,
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### Barlow Twins Training begins")

    device = torch.device(f"cuda:{device_id}")
    # model = model.to(device)

    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            with torch.cuda.amp.autocast():
                output = model(data)
                output_cap = model(data_cap)

                _, proj = output["features"], output["proj"]
                _, proj_cap = output_cap["features"], output_cap["proj"]

                loss_con = loss_base(proj, proj_cap)

            optimizer.zero_grad()
            scaler.scale(loss_con).backward()
            scaler.step(optimizer)
            scaler.update()       
            # loss_con.backward()
            # optimizer.step()
            opt_lr_schedular.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model