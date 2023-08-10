import torch
from torch.utils.data import TensorDataset
import numpy as np
import os
import pickle

from .federated_emnist import FederatedEMNISTDataset
from .fed_cifar100 import FederatedCIFAR100Dataset
from .reddit import RedditDataset
from .celeba import CelebADataset
from .partitioned_cifar10 import PartitionedCIFAR10Dataset
from .federated_emnist_iid import FederatedEMNISTDatasetIID
from .federated_emnist_noniid import FederatedEMNISTDataset_nonIID

from fedcor.utils import get_dataset

from .base_dataset import BaseDataset
from utils import logger

def load_data(args):
    if args.dataset in ["cifar", "mnist", "fmnist"]:
        
        args.num_users = args.total_num_clients
        args.alpha = args.dirichlet_alpha

        cache_path = os.path.join(args.data_dir,
            f"{args.dataset}_N{args.num_users}_"
            f"alpha{'-none' if args.alpha is None else args.alpha}_"
            f"{'iid' if args.iid else 'noniid'}_"
            f"{args.shards_per_client}shard-per-client_"
            f"{'eq' if not args.unequal else 'uneq'}.pickle")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                dataset = pickle.load(fp)[0]
            logger.info(f"Load cached dataset from {cache_path}")
            return dataset

        ### No cached data found
        train_dataset, test_dataset, user_groups, user_groups_test, \
            weights = get_dataset(args, seed=None)
            
        assert len(user_groups) == len(user_groups_test)

        train_data_local_dict, train_data_local_num_dict = {}, {}
        test_data_local_dict, test_data_local_num_dict = {}, {}
        for client_idx in range(args.num_users):
           
            data_idxs: set = user_groups[client_idx]
            data_x, data_y = zip(*[train_dataset[i] for i in data_idxs])
            local_data = TensorDataset(torch.stack(data_x), torch.Tensor(data_y))
            train_data_local_dict[client_idx] = local_data
            train_data_local_num_dict[client_idx] = len(data_x)
            
            data_idxs: set = user_groups_test[client_idx]
            data_x, data_y = zip(*[test_dataset[i] for i in data_idxs])
            local_data = TensorDataset(torch.stack(data_x), torch.Tensor(data_y))
            test_data_local_dict[client_idx] = local_data
            test_data_local_num_dict[client_idx] = len(data_x)
        
        dataset = BaseDataset()
        dataset.num_classes = 10
        dataset.dataset['train'] = {
            'data_sizes': train_data_local_num_dict,
            'data': train_data_local_dict,
        }
        dataset.dataset['test'] = {
            'data_sizes': test_data_local_num_dict,
            'data': test_data_local_dict,
        }

        os.makedirs(args.data_dir, exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump([dataset], fp)
            logger.info(f"Dump cached dataset at {cache_path}")

        return dataset
        
    elif args.dataset == 'Reddit':
        return RedditDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST':
        return FederatedEMNISTDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST_IID':
        return FederatedEMNISTDatasetIID(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST_nonIID':
        return FederatedEMNISTDataset_nonIID(args.data_dir, args)
    elif args.dataset == 'FedCIFAR100':
        return FederatedCIFAR100Dataset(args.data_dir, args)
    elif args.dataset == 'CelebA':
        return CelebADataset(args.data_dir, args)
    elif args.dataset == 'PartitionedCIFAR10':
        return PartitionedCIFAR10Dataset(args.data_dir, args)

__all__ = ['FederatedEMNISTDataset', 'FederatedEMNISTDatasetIID', 'FederatedEMNISTDataset_nonIID',
            'FederatedCIFAR100Dataset', 'PartitionedCIFAR10Dataset', 'RedditDataset', 'CelebADataset',
            'load_data']

