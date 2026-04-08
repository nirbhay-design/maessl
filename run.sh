#!/bin/bash 

CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --config configs/bt_clr.yaml --dataset cifar100 --save_path bt.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --opt SGD --wd 1e-4 --warmup_epochs 10 > logs/bt.clr.r18.c100.log &

CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --config configs/vicreg_clr.yaml --dataset cifar100 --save_path vicreg.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --wd 1e-6 --opt LARS --warmup_epochs 10 > logs/vicreg.clr.r18.c100 &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt_clr.yaml --dataset timg --save_path bt.clr.r18.timg.pth --gpu 2 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 10 --distributed > logs/bt.clr.r18.timg &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/vicreg_clr.yaml --dataset timg --save_path vicreg.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --opt LARS --warmup_epochs 10  --distributed --port 5192 > logs/vicreg.clr.r18.timg &