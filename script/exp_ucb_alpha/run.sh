#!/bin/bash

set -x
export PBFL_EXP_DATETIME=`date '+%Y%m%d-%H%M%S'`
export PBFL_EXP_DIR=save/test_ucb_alpha/${PBFL_EXP_DATETIME}
mkdir -p $PBFL_EXP_DIR
DATADIR=./data

#################################################
#              Hyper-parameters
#################################################
BATCH_SIZE=32
TRAIN_ROUND=800
TOTAL_CLIENT_NUM=3400
LOCAL_EP=5

#################################################
#              Configurations to test
#################################################
DATASETS=(FederatedEMNIST_nonIID)

# Distribution types
# DIST_TYPES=(one_shard two_shard dir)
DIST_TYPES=(one_shard)

METHODS=(PBFL)
# METHODS=(Random PBFL FedCor Pow-d)

# UCB_ALPHAS=(const_1bslash10 const_0 const_2 const_5)
# UCB_ALPHAS=(const_1bslash10)
UCB_ALPHAS=(linear_1bslash100 linear_1bslash250 linear_1bslash500 linear_1bslash1000)

#################################################
#              Run
#################################################
for dataset in ${DATASETS[@]}; do
for dist_type in ${DIST_TYPES[@]}; do
for ucb_alpha in ${UCB_ALPHAS[@]}; do
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
    if [[ ${dataset} == "cifar" || ${dataset} == "FederatedEMNIST_nonIID" ]]; then
        model="CNN"
    elif [[ ${dataset} == "fmnist" ]]; then
        model="MLP"
    else
        exit
    fi

    export PBFL_EXP_NAME="${PBFL_EXP_DIR}/${method}_policy-${dist_type}-${ucb_alpha}_ucb_alpha-${TOTAL_CLIENT_NUM}to${NUM_CLIENT_PER_ROUND}-${dataset}-${model}"
    # echo $FedCorArg
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
        --ucb_alpha=${ucb_alpha} \
        ${FedCorArg} \
        $@
done
done
done
done