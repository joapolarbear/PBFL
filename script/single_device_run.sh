#!/bin/bash


# python3 pbfl/main.py --dataset FederatedEMNIST --method GPFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10
# pbfl/main.py --dataset PartitionedCIFAR10 --model CNN -A 10 -K 100 --lr_local 0.001 -B 50 -R 1000 --method GPFL --comment ucb0to4/5
# pbfl/main.py --dataset FederatedEMNIST_nonIID --method GPFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 500
MODEL=CNN
# MODEL=RESNET18
BATCH_SIZE=10
TRAIN_ROUND=1000
NUM_CLIENT_PER_ROUND=1
TOTAL_CLIENT_NUM=1

# DATASET=FederatedEMNIST
# DATASET=PartitionedCIFAR10
DATASET=cifar

DATADIR=./data

METHODS=(Single)


for METHOD in ${METHODS[@]}; do
    python3 pbfl/main.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 0.01 \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} -d 10 \
        --method ${METHOD} \
        --data_dir ${DATADIR} \
        --iid 0 \
        --shards_per_client 1 \
        $@
done