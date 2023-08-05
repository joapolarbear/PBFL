import torch
import numpy as np
import os
import pickle

def _distribution_str(dist: list, max_width = 3):
    _max = max(dist)
    assert _max < 10 ** (max_width + 1)
    return "|".join(["_" * max_width if c == 0 else "_" * (max_width-len(str(c))) + str(c) for c in dist])

def _label_to_distribution(_labels):
    if isinstance(_labels, torch.Tensor):
        _labels = _labels.detach().numpy().astype(int)
    cnt = np.array(np.unique(_labels, return_counts=True)).T # of shape (N_sample, 2), values and its counts
    rst = np.zeros(np.max(_labels)+1, dtype=int)
    rst[cnt[:, 0]] = cnt[:, 1]
    return rst
    
class BaseDataset:
    ''' An example of dataset
    {
        "train": {
            "data_sizes": {client_id: dataset_size},
            "data": {client_id: TensorDataset(x, y)}
        },
        "test": {
            "data_sizes": {client_id: dataset_size},
            "data": {client_id: TensorDataset(x, y)}
        }
    }
    '''
    def __init__(self):
        self.num_classes = None
        self.dataset = {}
    
    def check_test_dist(self, name, dataset = None):
        _dataset = dataset or self.dataset
        test_data_local_dict: dict = _dataset["test"]["data"]
        labels_list = []
        for client_id, local_data in test_data_local_dict.items():
            x, y = local_data.tensors
            labels_list.append(y)
        lables = torch.cat(labels_list, dim=0)
        rst = _label_to_distribution(lables)
        print(f"{name}, len={len(lables)}, distribution: {_distribution_str(rst)}")

    def check_test_dist_by_client(self, name, dataset = None):
        _dataset = dataset or self.dataset
        test_data_local_dict: dict = _dataset["test"]["data"]
        for client_id, local_data in test_data_local_dict.items():
            x, y = local_data.tensors
            lables = [int(e) for e in y]
            rst = _label_to_distribution(lables)
            print(f"{name}_{client_id}, distribution: {_distribution_str(rst)}")

 

