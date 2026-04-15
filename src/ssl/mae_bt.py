import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributed as dist

def train_maebt(
        model, train_loader, loss_base, 
        optimizer, opt_lr_schedular, scaler, weight,
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### mae + bt Training begins")

    device = torch.device(f"cuda:{device_id}")
    # model = model.to(device)

    for epochs in range(n_epochs):
        if dist.is_initialized():
            # print(f'setting up epoch: {epochs}')
            train_loader.sampler.set_epoch(epochs)
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            # with torch.cuda.amp.autocast():
            data_combine = torch.cat([data, data_cap], dim = 0)
            output_combine = model(data_combine)

            proj, proj_cap = output_combine["proj"].chunk(2, dim = 0)

            loss_red = loss_base(proj, proj_cap)

            loss_con = output_combine["loss"] + weight * loss_red

            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()
            # scaler.scale(loss_con).backward()
            # scaler.step(optimizer)
            # scaler.update()       

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_red=loss_red.item(), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model