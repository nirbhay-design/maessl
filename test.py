import torch
import torch.nn as nn
import torch.optim as optim
from src.network import MLP, BaseEncoder
from train_utils import load_dataset, progress, yaml_loader, get_tsne_knn_logreg
import itertools
import argparse 
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description="Training script for linear probing")

    # basic experiment settings
    parser.add_argument("--dataset", type=str, default = "cifar10", required=True, help="dataset name")
    parser.add_argument("--saved_path", type=str, default="model.pth", required=True, help="path for pretrained model")
    parser.add_argument("--gpu", type=int, default = 0, help="gpu_id")
    parser.add_argument("--model", type=str, default="resnet18", help="resnet18/resnet50")
    parser.add_argument("--verbose", action="store_true", help="verbose or not")
    parser.add_argument("--epochs", type=int, default = 100, help="epochs for linear probing")
    parser.add_argument("--eval_every", type=int, default = 10, help="evaluation interval")
    parser.add_argument("--knn", action="store_true", help="evaluate knn or not")
    parser.add_argument("--lreg", action="store_true", help="evaluate logistic regression or not")
    parser.add_argument("--linprobe", action="store_true", help="evaluate linear probing or not ")
    parser.add_argument("--tsne", action="store_true", help="get test tsne or not")
    parser.add_argument("--umap", action="store_true", help="get test umap or not")
    parser.add_argument("--cmet", action="store_true", help="get clustering metrics or not")

    args = parser.parse_args()
    return args

def evaluate(model, mlp, loader, device, return_logs=False):
    model.eval()
    mlp.eval()
    correct = 0;samples =0
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            feats = model(x)
            scores = mlp(feats)

            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        accuracy = round(float(correct / samples), 3)
    return accuracy 

def train_mlp(
    model, mlp, train_loader, test_loader, 
    lossfunction, mlp_optimizer, n_epochs, eval_every,
    device_id, eval_id, return_logs=False, mlp_schedular=None):

    tval = {'trainacc':[],"trainloss":[], "testacc":[]}
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    mlp = mlp.to(device)
    for epochs in range(n_epochs):
        model.eval()
        mlp.train()
        curacc = 0
        cur_mlp_loss = 0
        len_train = len(train_loader)
        for idx , (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                feats = model(data)
            scores = mlp(feats.detach())      
            
            loss_sup = lossfunction(scores, target)

            mlp_optimizer.zero_grad()
            loss_sup.backward()
            mlp_optimizer.step()

            cur_mlp_loss += loss_sup.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_sup=loss_sup.item(), GPU = device_id)
        
        if mlp_schedular is not None:
            mlp_schedular.step()
        
        if epochs % eval_every == 0 and device_id == eval_id:
            cur_test_acc = evaluate(model, mlp, test_loader, device, return_logs)
            tval["testacc"].append(float(cur_test_acc))
            print(f"[GPU{device_id}] Test Accuracy at epoch: {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_mlp_loss))
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss_sup: {cur_mlp_loss:.3f}")
    
    if device_id == eval_id:
        final_test_acc = evaluate(model, mlp, test_loader, device, return_logs)
        tval["testacc"].append(float(final_test_acc))
        print(f"[GPU{device_id}] Final Test Accuracy: {final_test_acc}")

    return mlp, tval

def train_linear_probe(
    pretrain_model,
    train_loader, 
    test_loader, 
    num_classes, 
    device=0, 
    epochs=100,
    eval_every=10,
    return_logs=False
):
    # Standard sweep grids for linear probing
    learning_rates = [0.1, 1.0]
    weight_decays = [1e-6, 1e-4, 0.0]
    loss = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_hparams = {}

    print(f"Starting Hyperparameter Sweep on {device}...")
    for cosine in range(2):
        for lr, wd in itertools.product(learning_rates, weight_decays):
            mlp = MLP(pretrain_model.classifier_infeatures, num_classes, mlp_type = "linear").to(device)
            optimizer = optim.SGD(mlp.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

            if cosine:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            else:
                scheduler = None
            print(f"Scheduler: {scheduler}")
            print(f"MLP: {mlp}")

            mlp, tval = train_mlp(
                pretrain_model, mlp, train_loader, test_loader, 
                lossfunction=loss, mlp_optimizer=optimizer, n_epochs=epochs, 
                eval_every=eval_every, device_id=device, eval_id=device, return_logs=return_logs,
                mlp_schedular=scheduler
            )

            best_test_acc = max(tval['testacc'])

            print(f"LR: {lr:5.3f} | WD: {wd:7.6f} | Cosine: {bool(cosine)} | test Acc: {best_test_acc:.3f}%")
            
            if best_test_acc > best_acc:
                best_acc = best_test_acc
                best_hparams = {'lr': lr, 'wd': wd, "cosine": cosine}

    print("-" * 30)
    print(f"Best Test Accuracy: {best_acc:.3f}%")
    print(f"Optimal Hyperparameters: LR={best_hparams['lr']}, WD={best_hparams['wd']}, Cosine={best_hparams['cosine']}")

if __name__ == "__main__":
    args = get_args()
    print(args)

    config = yaml_loader("configs/test.yaml")

    encoder = BaseEncoder(model_name=args.model, pretrained=False)
    device = torch.device(f"cuda:{args.gpu}")
    print(encoder.load_state_dict(torch.load(args.saved_path, map_location=device)))

    _, train_dl_mlp, test_dl, _, _ = load_dataset(
        dataset_name = args.dataset,
        distributed = False,
        **config["dataset"][args.dataset]["params"])

    if args.linprobe:
        train_linear_probe(
            pretrain_model=encoder,
            train_loader=train_dl_mlp,
            test_loader=test_dl,
            num_classes=config["dataset"][args.dataset]["num_classes"],
            device=args.gpu,
            epochs=args.epochs,
            eval_every=args.eval_every,
            return_logs=args.verbose
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