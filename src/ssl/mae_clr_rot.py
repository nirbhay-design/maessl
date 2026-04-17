import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributed as dist

def train_maeclrrot(
        model, rotnet, train_loader, loss_base, save_model,
        optimizer, opt_lr_schedular, scaler, weight, # here weight is a dict
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### mae + simclr + rotnet Training begins")

    device = torch.device(f"cuda:{device_id}")
    # model = model.to(device)
    model.train()
    rotnet.train()

    for epochs in range(n_epochs):
        if dist.is_initialized():
            # print(f'setting up epoch: {epochs}')
            train_loader.sampler.set_epoch(epochs)
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, rot1, rot2, _, lrot1, lrot2) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            rot1 = rot1.to(device)
            rot2 = rot2.to(device)
            lrot1 = lrot1.to(device)
            lrot2 = lrot2.to(device)

            # with torch.cuda.amp.autocast():
            data_combine = torch.cat([data, data_cap], dim = 0)
            output_combine = model(data_combine)
            proj, proj_cap = output_combine["proj"].chunk(2, dim = 0)
            loss_clr = loss_base(proj, proj_cap)

            # evaluate rotnet output 
            combine_rot = torch.cat([rot1, rot2], dim = 0)
            combine_rot_label = torch.cat([lrot1, lrot2], dim = 0)
            combine_rot_out = rotnet(model(combine_rot, mask_ratio = 0.0)["features"])
            loss_rot = F.cross_entropy(combine_rot_out, combine_rot_label)

            loss_con = output_combine["loss"] + weight["wt1"] * loss_clr + weight["wt2"] * loss_rot

            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()
            # scaler.scale(loss_con).backward()
            # scaler.step(optimizer)
            # scaler.update()       

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_clr=loss_clr.item(), loss_con=loss_con.item(), loss_rot=loss_rot.item(), GPU = device_id)
        
        opt_lr_schedular.step()

        if (epochs + 1) % 100 == 0: # save every 100 epoch
            save_model(model, epochs = epochs + 1)

        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model