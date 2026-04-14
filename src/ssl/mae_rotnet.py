import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributed as dist 

class rotnet_cls(nn.Module):
    def __init__(self, in_features, rot_hidden, num_classes = 4):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(in_features, rot_hidden, bias=False),
                nn.BatchNorm1d(rot_hidden),
                nn.ReLU(),
                nn.Linear(rot_hidden, rot_hidden, bias=False),
                nn.BatchNorm1d(rot_hidden),
                nn.ReLU(),
                nn.Linear(rot_hidden, num_classes)
            )

    def forward(self, x):  
        return self.proj(x)

def train_maerotnet(
        model, rotnet, train_loader, # train_dl_mlp
        optimizer, opt_lr_schedular, scaler,
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### MAE + rotnet Training begins")

    device = torch.device(f"cuda:{device_id}")
    # model = model.to(device)
    model.train()
    rotnet.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, rot_data, _, rot_label) in enumerate(train_loader):
            data = data.to(device)
            rot_data = rot_data.to(device)
            rot_label = rot_label.to(device)

            output = model(data)
            output_rot = model(rot_data, mask_ratio = 0.0) # get the features for rotnet
            pred_rot = rotnet(output_rot["features"]) # labels for rotnet (predicted)
            
            loss_con = output["loss"] + 0.1 * F.cross_entropy(pred_rot, rot_label)

            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()
            # scaler.scale(loss_con).backward()
            # scaler.step(optimizer)
            # scaler.update()       

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model