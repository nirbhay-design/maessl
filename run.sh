#!/bin/bash 

# nohup python train.py --config configs/bt_clr.yaml --dataset cifar100 --save_path bt.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --opt LARS --warmup_epochs 10 --distributed > logs/bt.clr.r18.c100.log &

# nohup python train.py --config configs/vicreg_clr.yaml --dataset cifar100 --save_path vicreg.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --opt LARS --warmup_epochs 10  --distributed --port 8192 > logs/vicreg.clr.r18.c100 &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt_clr.yaml --dataset timg --save_path bt.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --opt LARS --warmup_epochs 10 --distributed > logs/bt.clr.r18.timg &

# CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/vicreg_clr.yaml --dataset timg --save_path vicreg.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.4 --opt LARS --warmup_epochs 10  --distributed --port 5192 > logs/vicreg.clr.r18.timg &

CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt.yaml --dataset cifar100 --save_path bt.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.8 --opt SGD --warmup_epochs 10 --distributed > logs/bt.r18.c100.log &
CUDA_VISIBLE_DEVICES=0,2 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt.yaml --dataset cifar100 --save_path bt.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.8 --opt SGD --warmup_epochs 0 --distributed > logs/bt.r18.c100.log &
