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

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset timg --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 256 --tbs 512 --save_path mae.clr.rot.timg.pth --model vit --nw 8 --wt 0.5 --wt2 0.5 > logs/mae.clr.rot.timg.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset img100 --gpu 2 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.clr.rot.i100.pth --model vit --nw 8 --wt 0.5 --wt2 0.5 > logs/mae.clr.rot.i100.log &

# nohup python -u train.py --config configs/mae_bt_rot.yaml --dataset timg --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.bt.rot.timg.pth --model vit --nw 8 --wt 0.1 --wt2 0.5 > logs/mae.bt.rot.timg.v1.log &

# nohup python -u train.py --config configs/mae_bt_rot.yaml --dataset img100 --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.bt.rot.i100.pth --model vit --nw 8 --wt 0.1 --wt2 0.5 > logs/mae.bt.rot.i100.log &


# ran it for wt 0.8 as well

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset timg --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.clr.rot.timg.wt0.1.pth --model vit --nw 8 --wt 0.1 --wt2 0.5 > logs/mae.clr.rot.timg.wt0.1.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset img100 --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.clr.rot.i100.wt0.1.pth --model vit --nw 8 --wt 0.1 --wt2 0.5 > logs/mae.clr.rot.i100.wt0.1.log &


# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_mask_rot.yaml --dataset img100 --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.mask.rot.i100.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.mask.rot.i100.wt0.5.log &

# CUDA_VISIBLE_DEVICES=0,1,2 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_mask_rot.yaml --dataset timg --gpu 2 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.mask.rot.timg.wt0.5.pth --model vit --nw 6 --wt 0.5 > logs/mae.mask.rot.timg.wt0.5.log &


# TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0,1,3,4 OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt_rot.yaml --dataset img100 --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 128 --tbs 512 --save_path mae.bt.rot.i100.pth --model vit --nw 4 --wt 0.1 --wt2 0.5 --distributed > logs/mae.bt.rot.i100.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset timg --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 256 --tbs 512 --save_path mae.clr.rot.timg.wt0.8wt20.1.pth --model vit --nw 10 --pf 2 --wt 0.8 --wt2 0.1 > logs/mae.clr.rot.timg.wt0.8wt20.1.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset img100 --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 256 --tbs 512 --save_path mae.clr.rot.i100.wt0.8wt20.1.pth --model vit --nw 10 --pf 2 --wt 0.8 --wt2 0.1 > logs/mae.clr.rot.i100.wt0.8wt20.1.log &


# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt_rot.yaml --dataset img100 --gpu 0 --epochs 400 --lr 5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.bt.rot.i100.lr5e.4.pth --model vit --nw 8 --pf 4 --wt 0.1 --wt2 0.5 --aug v2 > logs/mae.bt.rot.i100.lr5e.4.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt_rot.yaml --dataset timg --gpu 0 --epochs 400 --lr 5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.bt.rot.timg.lr5e.4.pth --model vit --nw 8 --pf 2 --wt 0.1 --wt2 0.5 --aug v2 > logs/mae.bt.rot.timg.lr5e.4.log &


# rotnet with dampening 

# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_clr_rot.yaml --dataset img100 --gpu 1 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 512 --tbs 512 --save_path mae.clr.rot.i100.wt0.8wt20.1.damp.pth --model vit --nw 6 --pf 2 --wt 0.8 --wt2 0.1 --damp_rot > logs/mae.clr.rot.i100.wt0.8wt20.1.damp.log &

###################### ** Test code ** ################################

# nohup python -u test.py --dataset img100 --saved_path saved_models/mae.i100.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet >> logs/mae.i100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.bt.i100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.bt.i100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.rot.i100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.rot.i100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.rot.timg.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.rot.timg.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.bt.i100.wt0.5.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.bt.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.bt.timg.wt0.5.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.bt.timg.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.clr.i100.wt0.5.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.clr.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.clr.timg.wt0.5.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 >> logs/mae.clr.timg.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.mask.rot.i100.wt0.5.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.mask.rot.i100.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.mask.rot.timg.wt0.5.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 6 >> logs/mae.mask.rot.timg.wt0.5.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.wt0.8.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 8 --pf 2 >> logs/mae.clr.rot.i100.wt0.8.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.wt0.8.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 8 --pf 2 >> logs/mae.clr.rot.timg.wt0.8.log &
 

## testing checkpoint models

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.ec200.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 5 >> checkpoint_logs/mae.clr.rot.timg.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.ec300.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 5 >> checkpoint_logs/mae.clr.rot.timg.ec300.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.ec200.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 5 >> checkpoint_logs/mae.clr.rot.i100.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.ec300.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 5 >> checkpoint_logs/mae.clr.rot.i100.ec300.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.ec200.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.timg.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.ec300.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.timg.ec300.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.ec200.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.i100.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.ec300.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.i100.ec300.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.wt0.8wt20.1.ec200.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.clr.rot.i100.wt0.8wt20.1.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.wt0.8wt20.1.ec200.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.clr.rot.timg.wt0.8wt20.1.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.wt0.8wt20.1.ec300.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.clr.rot.i100.wt0.8wt20.1.ec300.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.wt0.8wt20.1.ec300.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.clr.rot.timg.wt0.8wt20.1.ec300.log &


# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.wt0.8wt20.1.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> logs/mae.clr.rot.i100.wt0.8wt20.1.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.wt0.8wt20.1.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> logs/mae.clr.rot.timg.wt0.8wt20.1.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.clr.rot.i100.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.ec200.pth --gpu 4 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.clr.rot.i100.ec200.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.ec300.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.clr.rot.i100.ec300.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.i100.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.ec200.pth --gpu 7 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.i100.ec200.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.ec300.pth --gpu 7 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.i100.ec300.dist.log &

# TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.i100.dist.log &

# TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.pth --gpu 5 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.timg.dist.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.lr5e.4.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> logs/mae.bt.rot.i100.lr5e.4.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.lr5e.4.ec100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.i100.lr5e.4.ec100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.lr5e.4.ec200.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.i100.lr5e.4.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.lr5e.4.ec300.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.i100.lr5e.4.ec300.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.lr5e.4.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> logs/mae.bt.rot.timg.lr5e.4.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.lr5e.4.ec100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.timg.lr5e.4.ec100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.lr5e.4.ec200.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.timg.lr5e.4.ec200.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.lr5e.4.ec300.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 --lrs 5.0 10.0 15.0 20.0 30.0 >> checkpoint_logs/mae.bt.rot.timg.lr5e.4.ec300.log &


# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.ec100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.i100.ec100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.ec100.pth --gpu 1 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.bt.rot.timg.ec100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.wt0.8wt20.1.ec100.pth --gpu 2 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.clr.rot.i100.wt0.8wt20.1.ec100.log &

# OPENBLAS_NUM_THREADS=1 nohup python -u test_gs.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.wt0.8wt20.1.ec100.pth --gpu 0 --model vit --linprobe --lreg --knn --cmet --nw 4 --pf 2 >> checkpoint_logs/mae.clr.rot.timg.wt0.8wt20.1.ec100.log &



# OPENBLAS_NUM_THREADS=1 nohup python -u train.py --config configs/mae_bt_rot.yaml --dataset timg --gpu 0 --epochs 400 --lr 1.5e-4 --opt AdamW --wd 0.05 --warmup_epochs 20 --bs 256 --tbs 512 --save_path mae.bt.rot.timg.wt10.05.wt20.5.dist.pth --model vit --nw 4 --pf 3 --wt 0.05 --wt2 0.5 --distributed > logs/mae.bt.rot.timg.wt10.05.wt20.5.dist.log &



### TSNE plots for timg and img100

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.rot.timg.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.rot.i100.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset timg --saved_path saved_models/mae.timg.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset img100 --saved_path saved_models/mae.i100.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset timg --saved_path saved_models/mae.rot.timg.wt0.5.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset img100 --saved_path saved_models/mae.rot.i100.wt0.5.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset timg --saved_path saved_models/mae.clr.timg.wt0.5.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.i100.wt0.5.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset timg --saved_path saved_models/mae.bt.timg.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset img100 --saved_path saved_models/mae.bt.i100.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset timg --saved_path saved_models/mae.clr.rot.timg.wt0.8wt20.1.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose

# OPENBLAS_NUM_THREADS=1 python -u test_gs.py --dataset img100 --saved_path saved_models/mae.clr.rot.i100.wt0.8wt20.1.pth --gpu 0 --model vit --tsne --nw 4 --pf 2 --verbose


### attention maps visualization 

images=(
    # "datasets/imagenet100/train.X2/n01443537/n01443537_129.JPEG" 
    # "datasets/imagenet100/train.X3/n01601694/n01601694_21.JPEG" 
    # "datasets/imagenet100/train.X4/n01855672/n01855672_47.JPEG"
    # "datasets/imagenet100/train.X4/n01770393/n01770393_137.JPEG"
    # "datasets/imagenet100/train.X4/n01491361/n01491361_57.JPEG"
    # "datasets/imagenet100/train.X4/n01806143/n01806143_130.JPEG"
    # "datasets/imagenet100/train.X4/n01755581/n01755581_95.JPEG"
    # "datasets/imagenet100/train.X2/n01608432/n01608432_56.JPEG"
    "datasets/imagenet100/train.X1/n01820546/n01820546_27.JPEG"
)

models=(
    "saved_models/mae.clr.rot.i100.wt0.8wt20.1.pth" 
    "saved_models/mae.bt.rot.i100.pth" 
    "saved_models/mae.i100.pth")

for img in "${images[@]}"; do
    echo "Processing image: $img"
    
    # Loop through each model for the current image
    for mod in "${models[@]}"; do
        echo "  Running model: $mod"
        python attention_vis.py --saved_path "$mod" --gpu 0 --image "$img" --threshold 0.6
    done
    
    echo "Finished processing $img across all models."
    echo "--------------------------------------------------"
done

