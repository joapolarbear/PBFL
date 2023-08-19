#!/bin/bash


# python3 src/main.py --dataset FederatedEMNIST --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10
# src/main.py --dataset PartitionedCIFAR10 --model CNN -A 10 -K 100 --lr_local 0.001 -B 50 -R 1000 --method PBFL --comment ucb0to4/5
# src/main.py --dataset FederatedEMNIST_nonIID --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 500
MODEL=CNN
# MODEL=RESNET18
BATCH_SIZE=50
TRAIN_ROUND=1000
NUM_CLIENT_PER_ROUND=5
TOTAL_CLIENT_NUM=100
LOCAL_EP=5


# DATASET=FederatedEMNIST
# DATASET=PartitionedCIFAR10
DATASET=cifar


DATADIR=./data
# METHODS=(Random)
# METHODS=(PBFL)
# METHODS=(FedCor)
# METHODS=(Pow-d)
METHODS=(Random PBFL FedCor Pow-d)

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
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 0.01 --lr_decay=1.0 --wdecay=3e-4 \
        -E $LOCAL_EP \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} -d 10 \
        --method ${METHOD} \
        --data_dir ${DATADIR} \
        --iid 0 --unequal=0 \
        # --dirichlet_alpha 0.2 \
        --shards_per_client 1 \
        --warmup=20 \
        ${FedCorArg} \
        $@
done