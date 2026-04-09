import sys, random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.network import Network, EnergyScoreNet, BaseEncoder
from src.mae import * 
from src.ssl import pretrain_algo 
from train_utils import yaml_loader, model_optimizer, progress, format_time, \
                        loss_function, load_dataset, get_tsne_knn_logreg
from test import train_linear_probe
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import GradScaler
import torch.distributed as dist 
import os 
from functools import partial
import argparse
import json, time 

def get_args():
    parser = argparse.ArgumentParser(description="Training script")

    # basic experiment settings
    parser.add_argument("--config", type=str, default = "configs/nodel.c10.yaml", required=True, help="config file")
    parser.add_argument("--dataset", type=str, default = "cifar10", required=True, help="dataset name")
    parser.add_argument("--save_path", type=str, default="model.pth", required=True, help="path to save model")
    parser.add_argument("--gpu", type=int, default = 0, help="gpu_id")
    parser.add_argument("--model", type=str, default="resnet50", help="resnet18/resnet50/vit")
    parser.add_argument("--verbose", action="store_true", help="verbose or not")
    parser.add_argument("--epochs", type=int, default = None, help="epochs for SSL pretraining")
    parser.add_argument("--epochs_lin", type=int, default = None, help="epochs for linear probing")
    parser.add_argument("--opt", type=str, default=None, help="SGD/ADAM/AdamW/LARS")
    parser.add_argument("--lr", type=float, default = None, help="lr for SSL")
    parser.add_argument("--wd", type=float, default = None, help="weight decay for SSL")
    parser.add_argument("--warmup_epochs", type=int, default = None, help="warmup epochs before starting base_lr")
    parser.add_argument("--port", type=str, default = "4084", help="port to run distributed training")
    parser.add_argument("--distributed", action="store_true", help="distributed training")
    parser.add_argument("--bs", type=int, default = None, help="batch size per gpu")    
    parser.add_argument("--tbs", type=int, default = None, help="batch size per gpu for testing")    
    # evaluation 
    # parser.add_argument("--mlp_type", type=str, default=None, help="hidden/linear")
    parser.add_argument("--test", action="store_true", help="test or not")
    parser.add_argument("--knn", action="store_true", help="evaluate knn or not")
    parser.add_argument("--lreg", action="store_true", help="evaluate logistic regression or not")
    parser.add_argument("--linprobe", action="store_true", help="evaluate linear probing or not ")
    parser.add_argument("--tsne", action="store_true", help="get test tsne or not")

    args = parser.parse_args()
    return args

def ddp_setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    init_process_group(backend = 'nccl', rank = rank, world_size = world_size)

def train_network(**kwargs):
    train_algo = kwargs['train_algo']
    kwargs.pop("train_algo")
    model = pretrain_algo[train_algo](**kwargs)
    return model 

def main_single(rank=0, world_size=1, config={}, args=None, is_distributed=False):
    if is_distributed:
        ddp_setup(rank, world_size, args.port)

    torch.cuda.set_device(rank)

    device = config['gpu_id']
    train_algo = config['train_algo']

    if args.model == "vit":
        model = MaskedAutoencoderViT(**config["model_params"], norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(rank)
    else:
        model = Network(**config['model_params']).to(rank)
    if rank == device:
        print(model)
        print(f"NOC: {config['dataset'][args.dataset]['num_classes']}")

    train_dl, train_dl_mlp, _, train_ds, test_ds = load_dataset(
        dataset_name = args.dataset,
        distributed = is_distributed,
        **config["dataset"][args.dataset]["params"])

    optimizer = model_optimizer(model, config['opt'], **config['opt_params'])
    # config["schedular_params"]["T_max"] = config["schedular_params"]["T_max"] * len(train_dl)
    if config["warmup_epochs"] > 0:
        warmup_steps = config["warmup_epochs"] # len(train_dl) * config["warmup_epochs"]
        opt_lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])
        warmup_lr_schedular = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters = warmup_steps)
        schedular = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_schedular, opt_lr_schedular],
            milestones=[warmup_steps] 
        )
    else:
        schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])
    scaler = GradScaler()

    if rank == device:
        print(f"# of Training Images: {len(train_ds)}")
        print(f"# of Testing Images: {len(test_ds)}")
        print(f"schedular: {schedular}")
    
    loss_base = None # for instance for mae this will be none 
    if train_algo in ["bt_clr", "vicreg_clr"]:
        loss_clr = loss_function(loss_type = "simclr", **config.get('loss_params', {}).get("simclr", {}))
        base_algo = train_algo.split("_")[0]
        loss_base = loss_function(loss_type = base_algo, **config.get('loss_params', {}).get(base_algo, {}))
        if rank == device:
            print(f"loss: {loss_clr}")
            print(f"loss: {loss_base}")
    
    elif train_algo in ["bt", "mae_bt"]:
        loss_base = loss_function(loss_type = "bt", **config.get('loss_params', {}))

    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    return_logs = config['return_logs']
    # eval_every = config['eval_every']
    n_epochs = config['n_epochs']
    # n_epochs_mlp = config['n_epochs_mlp']

    # tsne_name = "_".join(config["model_save_path"].split('/')[-1].split('.')[:-1]) + ".png"
    
    ## defining parameter configs for each training algorithm

    param_config = {"train_algo": train_algo, "model": model, "train_loader": train_dl, "scaler": scaler,
        "loss_base": loss_base, "optimizer": optimizer, "opt_lr_schedular": schedular, "progress": progress,
        "n_epochs": n_epochs, "device_id": rank, "eval_id": device, "return_logs": return_logs}

    if train_algo in ["bt_clr", "vicreg_clr"]:
        param_config["loss_clr"] = loss_clr

    if train_algo in ["mae"]:
        print("using basic dataloader for MAE")
        param_config.pop("loss_base", -1) # not required for mae 
        param_config["train_loader"] = train_dl_mlp # this data loader is used for mae (less heavy augmentations)

    final_model = train_network(**param_config)

    if is_distributed:
        dist.barrier()
        print(f"rank:{rank} reached barrier")

    if rank == device:
        final_model = final_model.module if is_distributed else final_model 
        torch.save(final_model.base_encoder.state_dict(), config["model_save_path"])
        print("Model weights saved")

    if is_distributed:
        print(f"destroying for rank: {rank}")
        destroy_process_group() 

if __name__ == "__main__":
    # editing config based on arguments 

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    args = get_args()
    config = yaml_loader(args.config)

    config["config"] = args.config
    config['gpu_id'] = args.gpu
    config['model_params']['model_name'] = args.model
    config["return_logs"] = args.verbose
    config["model_save_path"] = os.path.join(config.get("model_save_path", "saved_models"), args.save_path)

    if args.bs:
        config["dataset"][args.dataset]["params"]["batch_size"] = args.bs
    if args.tbs: 
        config["dataset"][args.dataset]["params"]["test_batch_size"] = args.tbs
    if args.opt:
        config["opt"] = args.opt
        if args.opt in ["ADAM", "AdamW"]:
            config["opt_params"].pop("momentum", -1)
            config["opt_params"].pop("nesterov", -1)
            config["opt_params"]["betas"] = (0.9, 0.95) # for mae
    if args.lr:
        bs = config["dataset"][args.dataset]["params"]["batch_size"] # batch_size per gpu
        ws = torch.cuda.device_count() if args.distributed else 1.0
        config["opt_params"]["lr"] = args.lr * bs * ws / 256.0
    if args.wd:
        config["opt_params"]["weight_decay"] = args.wd
    if args.warmup_epochs is not None:
        config["warmup_epochs"] = args.warmup_epochs
    if args.epochs:
        config["n_epochs"] = args.epochs
        config["schedular_params"]["T_max"] = args.epochs - config["warmup_epochs"]
    if args.epochs_lin:
        config["n_epochs_mlp"] = args.epochs_lin
    if args.distributed:
        config["distributed"] = args.distributed
    
    # setting seeds 

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("environment: ")
    print(f"YAML: {args.config}")
    for key, value in config.items():
        print(f"==> {key}: {value}")

    print("-"*50)

    pt1 = time.perf_counter()
    # pretraining phase
    if args.distributed:
        world_size = torch.cuda.device_count()
        print(f"Launching DDP across {world_size} GPUs")
        mp.spawn(main_single, args=(world_size, config, args, True), nprocs=world_size)
    else:
        main_single(rank=args.gpu, world_size=1, config=config, args=args, is_distributed=False)
    pt2 = time.perf_counter()

    print("-"*50)
    # Running Linear probing 
    lpt1 = time.perf_counter()

    print("starting linear probing")
    if args.model == "vit":
        model_params = config["model_params"]
        encoder = MAEEncoder(img_size=model_params["img_size"], patch_size=model_params["patch_size"], in_chans=model_params["in_chans"],
                 embed_dim=model_params["embed_dim"], depth=model_params["depth"], num_heads=model_params["num_heads"],
                 mlp_ratio=model_params["mlp_ratio"], norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        encoder = BaseEncoder(model_name=args.model, pretrained=False)
    device = torch.device(f"cuda:{args.gpu}")
    print(encoder.load_state_dict(torch.load(config["model_save_path"], map_location=device)))

    _, train_dl_mlp, test_dl, _, _ = load_dataset(
        dataset_name = args.dataset,
        distributed = False,
        **config["dataset"][args.dataset]["params"])
    
    train_linear_probe(
        pretrain_model=encoder,
        train_loader=train_dl_mlp,
        test_loader=test_dl,
        num_classes=config["dataset"][args.dataset]["num_classes"],
        device=args.gpu,
        epochs=config["n_epochs_mlp"],
        eval_every=config['eval_every'],
        return_logs=args.verbose
    )
    
    tsne_name = ".".join(config["model_save_path"].split('/')[-1].split('.')[:-1]) + ".png"
    test_config = {"model": encoder, "train_loader": train_dl_mlp, "test_loader": test_dl, 
                    "device": device, "return_logs": args.verbose, "umap": False, "cmet": True,
                    "tsne": args.dataset == "cifar10", "knn": True, "log_reg": True, "tsne_name": tsne_name}
    
    output = get_tsne_knn_logreg(**test_config)
    for key, value in output.items():
        print(f"{key}: {value:.3f}")
    
    lpt2 = time.perf_counter()

    print(f"pretraining time: {format_time(pt2 - pt1)}")
    print(f"linear probing time: {format_time(lpt2 - lpt1)}")
    