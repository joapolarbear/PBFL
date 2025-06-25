#!/bin/bash
DROP_NUM=10
export PBFL_EXP_DATETIME=`date '+%Y%m%d-%H%M%S'`
export PBFL_EXP_DIR=save/results/${PBFL_EXP_DATETIME}-drop_num_${DROP_NUM}-3rdparty
mkdir -p $PBFL_EXP_DIR

dist_type=one_shard
TOTAL_CLIENT_NUM=100
NUM_CLIENT_PER_ROUND=10
dataset=fmnist
model=mlp

# random 1 shard fmnist
# method='random'
# export EXP_NAME_SHORT="${method}_policy-${dist_type}-${TOTAL_CLIENT_NUM}to${NUM_CLIENT_PER_ROUND}-${dataset}-${model}-3rdparty"
# export PBFL_EXP_PATH="${PBFL_EXP_DIR}/$EXP_NAME_SHORT"
# python3 3rdparty/FedCor/main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  --epochs=500 --num_user=$TOTAL_CLIENT_NUM \
#      --frac=0.1 --local_ep=3 --local_bs=64 --lr=5e-3 --schedule 150 300 --lr_decay=0.5 \
#      --optimizer=sgd --iid=0 --unequal=0  --shards_per_client=1 --verbose=1 --seed 1 2 3 4 5 \
#      --drop_client=1 \
#      2>&1 | tee ${PBFL_EXP_PATH}_log.txt


#### FedCor 1 shard fmnist
# method='fedcor'
# export EXP_NAME_SHORT="${method}_policy-${dist_type}-${TOTAL_CLIENT_NUM}to${NUM_CLIENT_PER_ROUND}-${dataset}-${model}-3rdparty"
# export PBFL_EXP_PATH="${PBFL_EXP_DIR}/$EXP_NAME_SHORT"
# python3 3rdparty/FedCor/main.py --gpu=0 --gpr_gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  \
#     --epochs=500 --num_user=$TOTAL_CLIENT_NUM --frac=0.1 --local_ep=3 --local_bs=64 --lr=5e-3 \
#     --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 \
#     --gpr --poly_norm=0 --GPR_interval=10 --group_size=100 --GPR_gamma=0.95 --update_mean --warmup=15 --discount=0.95 \
#     --shards_per_client=1 \
#     --drop_client=1 \
#     2>&1 | tee ${PBFL_EXP_PATH}_log.txt


#### Pow-d1 shard fmnist
method='powd'
export EXP_NAME_SHORT="${method}_policy-${dist_type}-${TOTAL_CLIENT_NUM}to${NUM_CLIENT_PER_ROUND}-${dataset}-${model}-3rdparty"
export PBFL_EXP_PATH="${PBFL_EXP_DIR}/$EXP_NAME_SHORT"
python3 3rdparty/FedCor/main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30 \
    --epochs=500 --num_user=$TOTAL_CLIENT_NUM --frac=0.05 --local_ep=3 --local_bs=64 --lr=5e-3 \
    --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 \
    --power_d --d=10 --shards_per_client=1 \
    --drop_client=1 \
    2>&1 | tee ${PBFL_EXP_PATH}_log.txt

# #### AFL 1 shard fmnist
# method='afl'
# export EXP_NAME_SHORT="${method}_policy-${dist_type}-${TOTAL_CLIENT_NUM}to${NUM_CLIENT_PER_ROUND}-${dataset}-${model}-3rdparty"
# export PBFL_EXP_PATH="${PBFL_EXP_DIR}/$EXP_NAME_SHORT"
# python3 3rdparty/FedCor/main.py --gpu=0 --dataset=fmnist --model=mlp --mlp_layer 64 30  \
#     --epochs=500 --num_user=$TOTAL_CLIENT_NUM --frac=0.05 --local_ep=3 --local_bs=64 --lr=5e-3 \
#     --schedule 150 300 --lr_decay=0.5 --optimizer=sgd --iid=0 --unequal=0  --verbose=1 --seed 1 2 3 4 5 \
#     --afl --shards_per_client=1 \
#     --drop_client=1 \
#     2>&1 | tee ${PBFL_EXP_PATH}_log.txt
