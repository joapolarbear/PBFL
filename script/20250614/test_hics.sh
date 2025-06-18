#!/bin/bash

set -x

DROP_NUM=1

export PBFL_EXP_DATETIME=`date '+%Y%m%d-%H%M%S'`
export PBFL_EXP_DIR=save/results/${PBFL_EXP_DATETIME}-hics
mkdir -p $PBFL_EXP_DIR
DATADIR=./data

#################################################
#              Hyper-parameters
#################################################
BATCH_SIZE=50
TRAIN_ROUND=2000
TOTAL_CLIENT_NUM=100
LOCAL_EP=5

#################################################
#              Configurations to test
#################################################
# DATASETS=(FederatedEMNIST)
# DATASETS=(PartitionedCIFAR10)
DATASETS=(cifar)
# DATASETS=(fmnist cifar)

# Distribution types
DIST_TYPES=(one_shard two_shard dir)
# DIST_TYPES=(dir)

# METHODS=(Random)
# METHODS=(GPFL Random)
# METHODS=(FedCor)
# METHODS=(Pow-d)
# METHODS=(DivFL)
METHODS=(HiCS)
# METHODS=(GPFL FedCor Random Pow-d DivFL)
# METHODS=(Random FedCor Pow-d)

#################################################
#              Run
#################################################
for dataset in ${DATASETS[@]}; do
for dist_type in ${DIST_TYPES[@]}; do
for method in ${METHODS[@]}; do

    if [[ ${dist_type} == "one_shard" ]]; then
        NUM_CLIENT_PER_ROUND=10
        DIST_ARG="--shards_per_client=1 "
    elif [[ ${dist_type} == "two_shard" ]]; then
        NUM_CLIENT_PER_ROUND=5
        DIST_ARG="--shards_per_client=2 "
    elif [[ ${dist_type} == "dir" ]]; then
        NUM_CLIENT_PER_ROUND=5
        DIST_ARG="--dirichlet_alpha=0.2 "
    else
        echo "Error"
        exit
    fi

    if [ $method = "FedCor" ]; then
        FedCorArg="--poly_norm=0 --update_mean  \
            --group_size=500 --GPR_gamma=0.99 \
            --discount=0.9 --GPR_interval=50"
    else  
        FedCorArg=""
    fi

    # MODELS=(RESNET18)
    if [[ ${dataset} == "cifar" ]]; then
        model="CNN"
    elif [[ ${dataset} == "fmnist" ]]; then
        model="MLP"
    else
        exit
    fi

    export EXP_NAME_SHORT="${method}_policy-${dist_type}-${TOTAL_CLIENT_NUM}to${NUM_CLIENT_PER_ROUND}-${dataset}-${model}"
    export PBFL_EXP_NAME="${PBFL_EXP_DIR}/$EXP_NAME_SHORT"
    # PYTHONPATH=
    python3 pbfl/main.py \
        --dataset ${dataset} \
        --model ${model} \
        --kernel_sizes 3 3 3 --num_filters 32 64 64 --mlp_layer 64 \
        -A ${NUM_CLIENT_PER_ROUND} \
        -K ${TOTAL_CLIENT_NUM} \
        --lr_local 0.01 --lr_decay=1 --wdecay=3e-4 \
        -E $LOCAL_EP \
        -B ${BATCH_SIZE} -R ${TRAIN_ROUND} -d 10 \
        --method ${method} \
        --data_dir ${DATADIR} \
        --gpu=0 \
        --iid=0 --unequal=0 \
        ${DIST_ARG} \
        --momentum=0 \
        --warmup=20 \
        ${FedCorArg} \
        $@
done
done
done
# --dirichlet_alpha=0.2 


#################################################
#              Backup
#################################################
# python3 pbfl/main.py --dataset FederatedEMNIST --method GPFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 200 -d 10
# pbfl/main.py --dataset PartitionedCIFAR10 --model CNN -A 10 -K 100 --lr_local 0.001 -B 50 -R 1000 --method GPFL --comment ucb0to4/5
# pbfl/main.py --dataset FederatedEMNIST_nonIID --method GPFL --model CNN -A 10 -K 200 --lr_local 0.01 -B 20 -R 500
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