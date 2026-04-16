#!/bin/bash 

# CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --config configs/bt_clr.yaml --dataset cifar100 --save_path bt.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --opt SGD --wd 1e-4 --warmup_epochs 10 > logs/bt.clr.r18.c100.log &

# CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --config configs/vicreg_clr.yaml --dataset cifar100 --save_path vicreg.clr.r18.c100.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --wd 1e-6 --opt LARS --warmup_epochs 10 > logs/vicreg.clr.r18.c100 &

# CUDA_VISIBLE_DEVICES=0,1 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/bt_clr.yaml --dataset timg --save_path bt.clr.r18.timg.pth --gpu 0 --model resnet18 --epochs 800 --lr 0.2 --opt SGD --wd 1e-4 --warmup_epochs 10 > logs/bt.clr.r18.timg &

# CUDA_VISIBLE_DEVICES=0,1 OPENBLAS_NUM_THREADS=1 nohup python train.py --config configs/vicreg_clr.yaml --dataset timg --save_path vicreg.clr.r18.timg.pth --gpu 1 --model resnet18 --epochs 800 --lr 0.2 --opt LARS --wd 1e-6 --warmup_epochs 10 > logs/vicreg.clr.r18.timg &


# nohup python train.py --config configs/mae_base.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 40 --bs 512 --tbs 512 --save_path mae.i100.pth --model vit > logs/mae.i100.log &

# nohup python train.py --config configs/mae_bt.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 40 --bs 512 --tbs 512 --save_path mae.bt.i100.pth --model vit > logs/mae.bt.i100.log &


# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_base.yaml --dataset timg --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.timg.pth --model vit --nw 6 > logs/mae.timg.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt.yaml --dataset timg --gpu 2 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.bt.timg.pth --model vit --nw 6 > logs/mae.bt.timg.log &

# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_rot.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.rot.i100.pth --model vit --nw 6 > logs/mae.rot.i100.log &

# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_rot.yaml --dataset timg --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.rot.timg.pth --model vit --nw 6 > logs/mae.rot.timg.log &

# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_rot.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.rot.i100.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.rot.i100.wt0.5.log &

# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_rot.yaml --dataset timg --gpu 2 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.rot.timg.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.rot.timg.wt0.5.log &


# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 40 --bs 512 --tbs 512 --save_path mae.bt.i100.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.bt.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt.yaml --dataset timg --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.bt.timg.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.bt.timg.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.clr.i100.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.clr.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr.yaml --dataset timg --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.clr.timg.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.clr.timg.wt0.5.log &


###################### ** Test code ** ################################

# nohup python -u test.py --dataset img100 --saved_path saved_models/mae.i100.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet >> logs/mae.i100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.bt.i100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.bt.i100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.rot.i100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.rot.i100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.rot.timg.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.rot.timg.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.bt.i100.wt0.5.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.bt.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.bt.timg.wt0.5.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.bt.timg.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.clr.i100.wt0.5.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.clr.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.clr.timg.wt0.5.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.clr.timg.wt0.5.log &