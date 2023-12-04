#!/bin/bash
export WANDB_LOG_PATH=wandb

# random  dir cifar
# python3 main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=100 \
#      --alpha=0.2 --frac=0.05 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 \
#      --optimizer=sgd --iid=0 --unequal=0 --verbose=1 --seed 1 2 3 4 5 \

# FedCor
python3 3rdparty/FedCor/main.py \
    --gpu=0 --gpr_gpu=0 \
    --dataset=cifar --model=cnn \
    --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
    --epochs=2000 --local_ep=5 \
    --num_user=100 \
    --frac=0.1 \
    --local_bs=50 \
    --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 \
    --iid 0 --unequal=0 \
    --shards_per_client 1 \
    --verbose=0 \
    --seed 1 2 3 4 5 \
    --gpr --discount=0.9 --GPR_interval=50 \
    --group_size=500 --GPR_gamma=0.99 \
    --poly_norm=0 --update_mean --warmup=20

# python3 3rdparty/FedCor/main.py \
#     --gpu=0 --dataset=cifar --model=cnn \
#     --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
#     --epochs=2000 --num_user=100 --frac=0.1 --local_ep=5 \
#     --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd \
#     --reg=3e-4 --iid=0 --unequal=0 --shards_per_client 1 \
#     --verbose=1 --seed 1 2 3 4 5


# python3 3rdparty/FedCor/main.py \
#     --gpu=0 --dataset=cifar --model=cnn \
#     --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
#     --epochs=2000 --num_user=100 --frac=0.1 --local_ep=5 \
#     --local_bs=50 --lr=0.01 --lr_decay=1.0 --optimizer=sgd \
#     --reg=3e-4 --iid=0 --unequal=0 --shards_per_client 1 \
#     --verbose=1 --seed 1 2 3 4 5 \
#     --power_d --d=10 
