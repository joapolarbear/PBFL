#!/bin/bash


# python3 src/main.py --dataset FederatedEMNIST --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10
# src/main.py --dataset PartitionedCIFAR10 --model CNN -A 10 -K 100 --lr_local 0.001 -B 50 -R 1000 --method PBFL --comment ucb0to4/5
# src/main.py --dataset FederatedEMNIST_nonIID --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 500
# FedCor
# python3 3rdparty/FedCor/main.py \
#     --gpu=0 --gpr_gpu=0 \
#     --dataset=cifar --model=cnn \
#     --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
#     --epochs=2000 --local_ep=5 \
#     --num_user=100 \
#     --frac=0.1 \
#     --local_bs=50 \
#     --lr=0.01 --lr_decay=1.0 --optimizer=sgd --reg=3e-4 \
#     --iid=0 --unequal=0 \
#     --shards_per_client 1 \
#     --verbose=0 \
#     --seed 1 2 3 4 5 \
#     --gpr --discount=0.9 --GPR_interval=50 \
#     --group_size=500 --GPR_gamma=0.99 \
#     --poly_norm=0 --update_mean --warmup=20
MODEL=CNN
# MODEL=RESNET18
BATCH_SIZE=50
TRAIN_ROUND=2000
NUM_CLIENT_PER_ROUND=5
TOTAL_CLIENT_NUM=100
LOCAL_EP=5


# DATASET=FederatedEMNIST
# DATASET=PartitionedCIFAR10
# DATASET=cifar
DATASET=cifar


DATADIR=./data
# METHODS=(Random)
METHODS=(PBFL)
# METHODS=(FedCor)
# METHODS=(Pow-d)
# METHODS=(Random PBFL FedCor Pow-d)

for METHOD in ${METHODS[@]}; do
    if [ $METHOD = "FedCor" ]; then
        FedCorArg="--poly_norm=0 --update_mean  \
            --group_size=500 --GPR_gamma=0.99 \
            --discount=0.9 --GPR_interval=50"
    else  
        FedCorArg=""
    fi
    # echo $FedCorArg
    python3 src/main.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 0.01 --lr_decay=1 --wdecay=3e-4 \
        -E $LOCAL_EP \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} -d 10 \
        --method ${METHOD} \
        --data_dir ${DATADIR} \
        --gpu=0 \
        --iid=0 --unequal=0 \
        --shards_per_client=2 \
        --warmup=20 \
        ${FedCorArg} \
        $@
done
# --dirichlet_alpha=0.2 