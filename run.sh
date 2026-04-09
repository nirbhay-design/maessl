#!/bin/bash 

# CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --config configs/bt_clr.yaml --dataset cifar100 --save_path bt.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --opt SGD --wd 1e-4 --warmup_epochs 10 > logs/bt.clr.r18.c100.log &

# CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --config configs/vicreg_clr.yaml --dataset cifar100 --save_path vicreg.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --wd 1e-6 --opt LARS --warmup_epochs 10 > logs/vicreg.clr.r18.c100 &

# CUDA_VISIBLE_DEVICES=0,1 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt_clr.yaml --dataset timg --save_path bt.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --opt SGD --wd 1e-4 --warmup_epochs 10 > logs/bt.clr.r18.timg &

# CUDA_VISIBLE_DEVICES=0,1 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/vicreg_clr.yaml --dataset timg --save_path vicreg.clr.r18.timg.pth --gpu 1 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 10 > logs/vicreg.clr.r18.timg &

nohup python train.py --config configs/mae_base.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 40 --bs 512 --tbs 512 --save_path mae.i100.pth --model vit > logs/mae.i100.log &
