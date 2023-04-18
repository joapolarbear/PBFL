from collections import Counter
import torch
from torch.utils.data import TensorDataset

from .federated_emnist import FederatedEMNISTDataset
from .fed_cifar100 import FederatedCIFAR100Dataset
from .reddit import RedditDataset
from .celeba import CelebADataset
from .partitioned_cifar10 import PartitionedCIFAR10Dataset
from .federated_emnist_iid import FederatedEMNISTDatasetIID
from .federated_emnist_noniid import FederatedEMNISTDataset_nonIID

def check_test_dist(name, dataset):
    _dataset = dataset.dataset
    test_data_local_dict: dict = _dataset["test"]["data"]
    labels_list = []
    for client_id, local_data in test_data_local_dict.items():
        x, y = local_data.tensors
        labels_list.append(y)
    lables = torch.cat(labels_list, dim=0)
    lables = [int(x) for x in lables]
    counter = Counter(lables)
    print(f"{name}, len={len(lables)}, distribution: {counter}")
    return counter

def check_test_dist_by_client(name, dataset):
    _dataset = dataset.dataset
    test_data_local_dict: dict = _dataset["test"]["data"]
    for client_id, local_data in test_data_local_dict.items():
        x, y = local_data.tensors
        lables = [int(x) for x in y]
        counter = Counter(lables)
        print(f"{name}_{client_id}, distribution: {counter}")

__all__ = ['FederatedEMNISTDataset', 'FederatedEMNISTDatasetIID', 'FederatedEMNISTDataset_nonIID',
            'FederatedCIFAR100Dataset', 'PartitionedCIFAR10Dataset', 'RedditDataset', 'CelebADataset',
            'check_test_dist', 'check_test_dist_by_client']
