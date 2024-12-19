# Federated Learning with Gradient Projection-Based Client Selection

This repository provides a framework for federated learning with various client selection methods, including our proposed Gradient Projection-Based Client Selection (GPFL). The experiments can be run on different datasets and models, with customizable hyperparameters.

## Requirements

Ensure you have the following dependencies installed:

```shell
torch=1.8.0
torchvision
numpy
scipy
tqdm
h5py
```

## Running Experiments

To run the experiments, execute the following command:

```shell
bash script/test.sh
```

The `test.sh` script runs federated learning experiments with different client selection methods on the specified dataset and model. Below are the details of the parameters used in the script.

### Parameters

- **MODEL**: The model architecture to be used. Options include `CNN` and `RESNET18`.
- **BATCH_SIZE**: The batch size for local training.
- **TRAIN_ROUND**: The number of training rounds.
- **NUM_CLIENT_PER_ROUND**: The number of clients selected per round.
- **TOTAL_CLIENT_NUM**: The total number of clients.
- **LOCAL_EP**: The number of local epochs.
- **DATASET**: The dataset to be used. Options include `FederatedEMNIST`, `PartitionedCIFAR10`, and `cifar`.
- **DATADIR**: The directory where the dataset is stored.
- **METHODS**: The client selection methods to be used. Options include `Random`, `GPFL`, `FedCorr`, and `Pow-d`.

### Example Command

The script runs the following command for each client selection method:

```shell
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
    --shards_per_client 1 \
    --dirichlet_alpha 0.2 \
    $@
```

### Client Selection Methods

 1. ```Random```: Random Selection
 2. ```Pow-d```: Power-of-d-Choice [[Yae Jee Cho et al., 2022](https://arxiv.org/pdf/2010.01243.pdf)]
 3. ```FedCor```: Diverse Client Selection for FL [[Tang, Minxue, et al., 20222](https://openaccess.thecvf.com/
 content/CVPR2022/papers/
 Tang_FedCor_Correlation-Based_Active_Client_Selection_Strategy_for_Heterogeneous_Federated_Learning_CVPR_2022_
 paper.pdf)]
 4. ```GPFL``` : our proposed Gradient Projection-Based Client Selection method

### Benchmark Datasets

To start training on different datasets with various client selection strategies, use the following datasets:

1. **FederatedEMNIST**
2. **PartitionedCIFAR10**
3. **cifar**

### Example Usage

To run an experiment with the `GPFL` method on the `cifar` dataset using the `CNN` model, you can use the following command:

```shell
python3 src/main.py \
    --dataset cifar \
    --model CNN \
    -A 5 \
    -K 100 \
    --lr_local 0.01 --lr_decay=1.0 --wdecay=3e-4 \
    -E 5 \
    -B 50 -R 1000 -d 10 \
    --method GPFL \
    --data_dir ./data \
    --iid 0 --unequal=0 \
    --shards_per_client 1 \
    --dirichlet_alpha 0.2
```

### References

- [FedML-AI](https://github.com/FedML-AI/FedML)
- [Accenture Labs Federated Learning](https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling)

This setup ensures efficient and reliable execution of federated learning experiments, leveraging high-performance hardware and up-to-date software libraries.
