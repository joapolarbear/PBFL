#!/bin/bash

# python3 src/main.py --dataset FederatedEMNIST --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10

MODEL=CNN
BATCH_SIZE=10
TRAIN_ROUND=1000
NUM_CLIENT_PER_ROUND=10
TOTAL_CLIENT_NUM=200
DATASET=FederatedEMNIST

METHODS=(FedCor Random DivFL)
METHODS=(FedCor)

for METHOD in ${METHODS[@]}; do
    python3 src/main.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 0.01 \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} \
        --method ${METHOD}
done