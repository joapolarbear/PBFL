#!/bin/bash

MODEL=MLP
# MODEL=RESNET18
BATCH_SIZE=64
TRAIN_ROUND=2000
NUM_CLIENT_PER_ROUND=5
TOTAL_CLIENT_NUM=100
LOCAL_EP=3

DATASET=fmnist


DATADIR=./data
METHODS=(FedCor)

for METHOD in ${METHODS[@]}; do
    python3 pbfl/main.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 5e-3 --lr_decay=0.5 --wdecay=1e-4 \
        -E $LOCAL_EP \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} -d 10 \
        --method ${METHOD} \
        --data_dir ${DATADIR} \
        --iid 0 --unequal=0 \
        --dirichlet_alpha=0.2 \
        --poly_norm=0 --update_mean --warmup=20 \
        --group_size=500 --GPR_gamma=0.99 \
        --discount=0.9 --GPR_interval=50 \
        $@
done

        # --shards_per_client 1 \

# CURDIRNAME=$(dirname $0)
# export WANDB_LOG_PATH=$(realpath $CURDIRNAME/../../)
# echo WANDB log path $WANDB_LOG_PATH/wandb

# python3 3rdparty/FedCor/main.py \
#     --gpu=0 --gpr_gpu=0 \
#     --dataset=cifar --model=cnn \
#     --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
#     --epochs=2000 --local_ep=5 \
#     --num_user=100 \
#     --frac=0.05 \
#     --local_bs=50 \
#     --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 \
#     --iid=0 --unequal=0 \
#     --alpha=0.2 \
#     --verbose=0 \
#     --seed 1 2 3 4 5 \
#     --gpr --discount=0.9 --GPR_interval=50 \
#     --group_size=500 --GPR_gamma=0.99 \
#     --poly_norm=0 --update_mean --warmup=20

