#!/bin/bash


# python3 src/main.py --dataset FederatedEMNIST --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10
# src/main.py --dataset PartitionedCIFAR10 --model CNN -A 10 -K 100 --lr_local 0.001 -B 50 -R 1000 --method PBFL --comment ucb0to4/5
# src/main.py --dataset FederatedEMNIST_nonIID --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 500
MODEL=CNN
# MODEL=RESNET18
BATCH_SIZE=10
TRAIN_ROUND=500
NUM_CLIENT_PER_ROUND=5
TOTAL_CLIENT_NUM=200

# DATASET=FederatedEMNIST
# DATASET=PartitionedCIFAR10
DATASET=cifar


DATADIR=./data
# METHODS=(Random)
# METHODS=(PBFL)
# METHODS=(FedCor )
# METHODS=(Pow-d)
METHODS=(Random PBFL FedCor Pow-d)

for METHOD in ${METHODS[@]}; do
    python3 src/main.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 0.01 \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} -d 10 \
        --method ${METHOD} \
        --data_dir ${DATADIR} \
        --iid 0 \
        --shards_per_client 2
done