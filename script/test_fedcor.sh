#!/bin/bash

CURDIRNAME=$(dirname $0)
export WANDB_LOG_PATH=$(realpath $CURDIRNAME/../../)
echo WANDB log path $WANDB_LOG_PATH/wandb

python3 3rdparty/FedCor/main.py \
    --gpu=0 --gpr_gpu=0 \
    --dataset=cifar --model=cnn \
    --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
    --epochs=2000 --local_ep=5 \
    --num_user=100 \
    --frac=0.05 \
    --local_bs=50 \
    --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 \
    --iid=0 --unequal=0 \
    --alpha=0.2 \
    --verbose=0 \
    --seed 1 2 3 4 5 \
    --gpr --discount=0.9 --GPR_interval=50 \
    --group_size=500 --GPR_gamma=0.99 \
    --poly_norm=0 --update_mean --warmup=20