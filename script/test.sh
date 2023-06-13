#!/bin/bash

# python3 src/main.py --dataset FederatedEMNIST --method PBFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10

python3 src/main.py \
    --dataset FederatedEMNIST \
    --model CNN -A 10 -K 200 \
    --lr_local 0.01 \
    -B 10 -R 500 \
    --method PBFL

python3 src/main.py \
    --dataset FederatedEMNIST \
    --model CNN -A 10 -K 200 \
    --lr_local 0.01 \
    -B 10 -R 500 \
    --method DivFL