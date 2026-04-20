import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributed as dist 

def train_mae_maskrotnet(
        model, rotnet, train_loader, save_model, # train_dl_mlp
        optimizer, opt_lr_schedular, scaler, weight,
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### MAE + Mask rotnet Training begins")

    device = torch.device(f"cuda:{device_id}")
    # model = model.to(device)
    model.train()
    rotnet.train()
    for epochs in range(n_epochs):
        if dist.is_initialized():
            # print(f'setting up epoch: {epochs}')
            train_loader.sampler.set_epoch(epochs)
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, _, rot_label) in enumerate(train_loader):
            data = data.to(device)
            rot_label = rot_label.to(device)

            output = model(data) # masked data (data is rotated)
            pred_rot = rotnet(output["features"][:, 1:, :].mean(dim = 1)) # labels for rotnet (predicted) for pooled features
            
            loss_con = output["loss"] + weight * F.cross_entropy(pred_rot, rot_label)

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

        if device_id == eval_id:
            if (epochs + 1) % 100 == 0 and (epochs + 1) < n_epochs: # save every 100 epoch
                save_model(model, epochs = epochs + 1)
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model