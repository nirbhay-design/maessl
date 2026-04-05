#!/bin/bash 

# CUDA_VISIBLE_DEVICES=2,3 nohup python train.py --config configs/bt_clr.yaml --dataset cifar100 --save_path bt.clr.r18.c100.pth --gpu 2 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 10 --distributed > logs/bt.clr.r18.c100.log &

# CUDA_VISIBLE_DEVICES=2,3 nohup python train.py --config configs/vicreg_clr.yaml --dataset cifar100 --save_path vicreg.clr.r18.c100.pth --gpu 3 --model resnet18 --epochs 800 --lr 0.2 --wd 1e-6 --opt LARS --warmup_epochs 10  --distributed --port 8192 > logs/vicreg.clr.r18.c100 &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt_clr.yaml --dataset timg --save_path bt.clr.r18.timg.pth --gpu 2 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 10 --distributed > logs/bt.clr.r18.timg &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/vicreg_clr.yaml --dataset timg --save_path vicreg.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --opt LARS --warmup_epochs 10  --distributed --port 5192 > logs/vicreg.clr.r18.timg &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt.yaml --dataset cifar100 --save_path bt.r18.c100.pth --gpu 2 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 10 --distributed --port 8192 > logs/bt.r18.c100.log &

# OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt.yaml --dataset cifar100 --save_path bt.r18.c100.we0.pth --gpu 2 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 0 > logs/bt.r18.c100.we0.log &
