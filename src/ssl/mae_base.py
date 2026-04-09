import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

def train_mae(
        model, train_loader, # train_dl_mlp
        optimizer, opt_lr_schedular, scaler,
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### MAE Training begins")

    device = torch.device(f"cuda:{device_id}")
    # model = model.to(device)

    model.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, _) in enumerate(train_loader):
            data = data.to(device)

            output = model(data)
            
            loss_con = output["loss"]

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