import torch
import torch.nn as nn
import torch.optim as optim
from src.network import MLP, BaseEncoder
from src.mae import MAEEncoder
from train_utils import load_dataset, progress, yaml_loader, get_tsne_knn_logreg, format_time
import itertools
import argparse 
from functools import partial
import torch.nn.functional as F
import time

def get_args():
    parser = argparse.ArgumentParser(description="Training script for linear probing")

    # basic experiment settings
    parser.add_argument("--dataset", type=str, default = "cifar10", required=True, help="dataset name")
    parser.add_argument("--saved_path", type=str, default="model.pth", required=True, help="path for pretrained model")
    parser.add_argument("--gpu", type=int, default = 0, help="gpu_id")
    parser.add_argument("--model", type=str, default="resnet18", help="resnet18/resnet50/vit")
    parser.add_argument("--verbose", action="store_true", help="verbose or not")
    parser.add_argument("--epochs", type=int, default = 100, help="epochs for linear probing")
    parser.add_argument("--eval_every", type=int, default = 10, help="evaluation interval")
    parser.add_argument("--knn", action="store_true", help="evaluate knn or not")
    parser.add_argument("--lreg", action="store_true", help="evaluate logistic regression or not")
    parser.add_argument("--linprobe", action="store_true", help="evaluate linear probing or not ")
    parser.add_argument("--tsne", action="store_true", help="get test tsne or not")
    parser.add_argument("--umap", action="store_true", help="get test umap or not")
    parser.add_argument("--cmet", action="store_true", help="get clustering metrics or not")
    parser.add_argument("--nw", type=int, default = 4, help="num workers for dataloading")
    parser.add_argument("--pf", type=int, default = 4, help="prefetch factor for dataloading")
    parser.add_argument("--aug", type=str, default = "v1", help="augmentation strategy")
    parser.add_argument("--lrs", type=list, default = [1.0, 1.5, 2.0, 5.0, 10.0], help="learning rates for grid search")

    args = parser.parse_args()
    return args

def evaluate(model, linear_probes, loader, device, return_logs=False):
    nlp = len(linear_probes)
    model.eval()
    for i in range(nlp):
        linear_probes[i]["mlp"].eval()

    correct_probes = [0 for _ in range(nlp)]
    samples_probes = [0 for _ in range(nlp)]

    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            feats = model(x)

            for i in range(nlp):
                scores = linear_probes[i]["mlp"](feats)
                _,predictions = scores.max(1)

                correct_probes[i] += (predictions == y).sum()
                samples_probes[i] += predictions.size(0)

            if return_logs:
                progress(idx+1,loader_len)

        accuracy_probes = [round(float(correct_probes[i] / samples_probes[i]), 3) for i in range(nlp)]  

    return accuracy_probes

def train_mlp(
    model, linear_probes, train_loader, test_loader, 
    lossfunction, n_epochs, eval_every,
    device_id, eval_id, return_logs=False):
    
    nlp = len(linear_probes)
    tval = [{'trainacc':[],"trainloss":[], "testacc":[]} for _ in range(nlp)]
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)

    for epochs in range(n_epochs):
        model.eval()
        for i in range(nlp):
            linear_probes[i]["mlp"].train()
        curacc = [0 for _ in range(nlp)]
        cur_mlp_loss = [0 for _ in range(nlp)]
        len_train = len(train_loader)
        for idx , (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                feats = model(data)

            lossp = {}
            for i in range(nlp):
                mlp_optimizer = linear_probes[i]["optimizer"]
                scores = linear_probes[i]["mlp"](feats.detach())
                loss_sup = lossfunction(scores, target)

                mlp_optimizer.zero_grad()
                loss_sup.backward()
                mlp_optimizer.step()

                cur_mlp_loss[i] += loss_sup.item() / (len_train)
                # scores = F.softmax(scores,dim = 1)
                _,predicted = torch.max(scores,dim = 1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                curacc[i] += correct / (samples * len_train)

                lossp[f"lp{i}"] = loss_sup.item()
            
            if return_logs:
                progress(idx+1,len(train_loader), GPU = device_id)
        
        for i in range(nlp):
            mlp_schedular = linear_probes[i]["scheduler"]
            if mlp_schedular is not None:
                mlp_schedular.step()
        
        if epochs % eval_every == 0 and device_id == eval_id:
            cur_test_acc = evaluate(model, linear_probes, test_loader, device, return_logs)
            print("--------------------------------")
            for i in range(nlp):
                tval[i]["testacc"].append(float(cur_test_acc[i]))
                print(f"[GPU{device_id}] Test Accuracy for probe{i} at epoch: {epochs}: {cur_test_acc[i]}")
            print("--------------------------------")

        print("--------------------------------")
        for i in range(nlp):
            tval[i]['trainacc'].append(float(curacc[i]))
            tval[i]['trainloss'].append(float(cur_mlp_loss[i]))
            print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc[i]:.3f} train_loss_sup: {cur_mlp_loss[i]:.3f}")
        print("--------------------------------")

    if device_id == eval_id:
        print("--------------------------------")
        final_test_acc = evaluate(model, linear_probes, test_loader, device, return_logs)
        for i in range(nlp):
            tval[i]["testacc"].append(float(final_test_acc[i]))
            print(f"[GPU{device_id}] Final Test Accuracy: {final_test_acc[i]}")
        print("--------------------------------")

    return linear_probes, tval

def train_linear_probe(
    pretrain_model,
    train_loader, 
    test_loader, 
    num_classes, 
    device=0, 
    epochs=100,
    eval_every=10,
    return_logs=False,
    learning_rates = [1.0, 1.5, 2.0, 5.0, 10.0]
):
    # Standard sweep grids for linear probing
    # learning_rates = [0.1, 0.7, 1.0, 1.5, 2.0]
    weight_decays = [1e-6, 1e-4, 0.0]
    loss = nn.CrossEntropyLoss()

    print(f"sweeping through lr: {learning_rates}")
    print(f"sweeping through wd: {weight_decays}")
    
    best_acc = 0.0
    best_hparams = {}
    linear_probes = []

    print(f"Starting Hyperparameter Sweep on {device}...")
    for cosine in range(2):
        for lr, wd in itertools.product(learning_rates, weight_decays):
            mlp = MLP(pretrain_model.classifier_infeatures, num_classes, mlp_type = "linear").to(device)
            optimizer = optim.SGD(mlp.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

            if cosine:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            else:
                scheduler = None

            linear_probes.append({
                "mlp": mlp,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "hparams": {"lr": lr, "wd": wd, "cosine": bool(cosine)}
            })
            print(f"Scheduler: {scheduler}")
            print(f"MLP: {mlp}")
            print(f"Optimizer: {optimizer}")

    linear_probes, tval = train_mlp(
        pretrain_model, linear_probes, train_loader, test_loader, 
        lossfunction=loss, n_epochs=epochs, eval_every=eval_every, 
        device_id=device, eval_id=device, return_logs=return_logs,
    )

    print("--------------------------------")
    for i in range(len(linear_probes)):
        best_test_acc = max(tval[i]['testacc'])
        lr = linear_probes[i]["hparams"]["lr"]
        wd = linear_probes[i]["hparams"]["wd"]
        cosine = linear_probes[i]["hparams"]["cosine"]
        print(f"LR: {lr:5.3f} | WD: {wd:7.6f} | Cosine: {cosine} | test Acc: {best_test_acc:.3f}%")
        
        if best_test_acc > best_acc:
            best_acc = best_test_acc
            best_hparams = {'lr': lr, 'wd': wd, "cosine": cosine}
    print("--------------------------------")
    

    print("-" * 30)
    print(f"Best Test Accuracy: {best_acc:.3f}%")
    print(f"Optimal Hyperparameters: LR={best_hparams['lr']}, WD={best_hparams['wd']}, Cosine={best_hparams['cosine']}")

if __name__ == "__main__":
    args = get_args()
    print(args)

    pt1 = time.perf_counter()

    config = yaml_loader("configs/test.yaml")
    config["dataset"][args.dataset]["params"]["num_workers"] = args.nw # set the number of workers for data loading 
    config["dataset"][args.dataset]["params"]["prefetch_factor"] = args.pf

    if args.model == "vit":
        model_params = config["mae_model_params"]
        if args.dataset == "timg":
            model_params["img_size"] = 64
            model_params["patch_size"] = 4
        encoder = MAEEncoder(img_size=model_params["img_size"], patch_size=model_params["patch_size"], in_chans=model_params["in_chans"],
                 embed_dim=model_params["embed_dim"], depth=model_params["depth"], num_heads=model_params["num_heads"],
                 mlp_ratio=model_params["mlp_ratio"], norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        encoder = BaseEncoder(model_name=args.model, pretrained=False)
    device = torch.device(f"cuda:{args.gpu}")
    print(encoder.load_state_dict(torch.load(args.saved_path, map_location=device)))
    encoder = encoder.to(device)

    dataloaders = load_dataset(
        dataset_name = args.dataset,
        distributed = False,
        aug = args.aug,
        **config["dataset"][args.dataset]["params"])
    
    train_dl_mlp = dataloaders.get("train_dl_mlp", None)
    test_dl = dataloaders.get("test_dl", None)

    if args.linprobe:
        train_linear_probe(
            pretrain_model=encoder,
            train_loader=train_dl_mlp,
            test_loader=test_dl,
            num_classes=config["dataset"][args.dataset]["num_classes"],
            device=args.gpu,
            epochs=args.epochs,
            eval_every=args.eval_every,
            return_logs=args.verbose,
            learning_rates=args.lrs
        )
    
    tsne_name = ".".join(args.saved_path.split("/")[-1].split('.')[:-1]) + '.png'
    test_config = {"model": encoder, "train_loader": train_dl_mlp, "test_loader": test_dl, 
                    "device": device, "return_logs": args.verbose, "umap": args.umap, "cmet": args.cmet,
                    "tsne": args.tsne, "knn": args.knn, "log_reg": args.lreg, "tsne_name": tsne_name}
    
    if any([args.tsne, args.knn, args.lreg, args.umap, args.cmet]):
        output = get_tsne_knn_logreg(**test_config)
        for key, value in output.items():
            print(f"{key}: {value:.3f}")
        # print(f"knn_acc: {output.get('knn_acc', -1):.3f}, log_reg_acc: {output.get('lreg_acc', -1):.3f}")

    pt2 = time.perf_counter()
    print(f"linear probing time: {format_time(pt2 - pt1)}")
