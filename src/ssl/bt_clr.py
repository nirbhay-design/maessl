import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

def train_btclr(
        model, train_loader, loss_base, loss_clr, 
        optimizer, opt_lr_schedular, scaler,
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    
    if device_id == eval_id:
        print(f"### Barlow Twins + SimCLR Training begins")

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

                _, proj_clr, proj_other = output["features"], output["proj_clr"], output["proj_other"]
                _, proj_clr_cap, proj_other_cap = output_cap["features"], output_cap["proj_clr"], output_cap["proj_other"]

                loss_simclr = loss_clr(proj_clr, proj_clr_cap)
                loss_red = loss_base(proj_other, proj_other_cap)

                loss_con = loss_red + 0.1 * loss_simclr

            optimizer.zero_grad()
            scaler.scale(loss_con).backward()
            scaler.step(optimizer)
            scaler.update()       
            opt_lr_schedular.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_simclr=loss_simclr.item(), loss_red=loss_red.item(), loss_con=loss_con.item(), GPU = device_id)
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model