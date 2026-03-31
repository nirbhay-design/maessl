#!/bin/bash 

# nohup python train.py --config configs/bt.yaml --dataset cifar100 --save_path bt.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --warmup_epochs 40 --distributed > logs/bt.clr.r18.c100.log &

# nohup python train.py --config configs/vicreg.yaml --dataset cifar100 --save_path vicreg.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --opt LARS --warmup_epochs 40 --lr 0.4 --distributed > logs/vicreg.clr.r18.c100 &

# OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt.yaml --dataset timg --save_path bt.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --warmup_epochs 40 --distributed > logs/bt.clr.r18.timg &

# OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/vicreg.yaml --dataset timg --save_path vicreg.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --opt LARS --warmup_epochs 40 --lr 0.4 --distributed > logs/vicreg.clr.r18.timg &