import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import json

from utils import logger

from .base_dataset import BaseDataset

class FederatedEMNISTDataset_nonIID(BaseDataset):
    def __init__(self, data_dir, args):
        '''
        known class: digits (10)
        unknown class: characters (52) -> label noise
        '''
        super(FederatedEMNISTDataset_nonIID, self).__init__()
        self.num_classes = 10
        self.min_num_samples = 10

        self._init_data(data_dir)
        logger.info(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')
        # 3383

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_nonIID.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess(data_dir, self.min_num_samples)
            
            # with open(file_name, 'wb') as f:
            #     pickle.dump(dataset, f)
            
        self.dataset = dataset


def _load_data(_dir):
    # Load the JSON file
    with open(_dir, 'r') as f:
        _data = json.load(f)
    # Convert the data to NumPy arrays
    num_samples = np.array(_data['num_samples']) # 
    user_names = np.array(_data['users']) # user_names corresponding to the array of num_samples
    _user2data = _data['user_data'] # {user_name: {'x': x, 'y': y}}
    _ids = list(_data['user_data'].keys())
    _num_clients = len(_ids)
    return _user2data, _ids, _num_clients

def _register_data(_user_data: dict, _data_local_dict: dict, _data_local_num_dict: dict,
                               client_idx: int, client_name: str, new_idx: int, min_num_samples: int, is_train: bool):
    # import pdb; pdb.set_trace()
    data_x = np.expand_dims(_user_data[client_name]['x'], axis=1)
    data_y = _user_data[client_name]['y']

    if is_train:
        if len(data_x) < min_num_samples:
            return False
        data_x = data_x[:min_num_samples]
        data_y = data_y[:min_num_samples]

    dim1, dim2, _ = data_x.shape
    data_x = data_x.reshape(dim1, dim2, 28, 28)
    local_data = TensorDataset(torch.Tensor(data_x), torch.Tensor(data_y))
    _data_local_dict[new_idx] = local_data
    _data_local_num_dict[new_idx] = len(data_x)
    
    if not is_train and len(data_x) == 0:
        logger.info(f"No test data for client {client_idx}")
    
    return True

def preprocess(data_dir, min_num_samples):
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    if True:
        train_data, train_ids, num_clients_train = _load_data('../dataset/FederatedEMNIST/mytrain.json')
        test_data, test_ids, num_clients_test = _load_data('../dataset/FederatedEMNIST/mytest.json')

        logger.info(f'#TrainClients {num_clients_train} #TestCli`ents {num_clients_test}')

        idx = 0
        for client_idx, client_name in enumerate(train_ids):
            succ = _register_data(train_data, train_data_local_dict, train_data_local_num_dict,
                client_idx, client_name, idx, min_num_samples, is_train=True)
            if not succ:
                continue
            _register_data(test_data, test_data_local_dict, test_data_local_num_dict,
                client_idx, client_name, idx, min_num_samples, is_train=False)
            idx += 1
    else:
        train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
        test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')
        # train_data = h5py.File(os.path.join(data_dir, 'mytrain.json'), 'r')
        # test_data = h5py.File(os.path.join(data_dir, 'mytest.json'), 'r')

        train_ids = list(train_data['examples'].keys())
        test_ids = list(test_data['examples'].keys())
        num_clients_train = len(train_ids) if num_clients is None else num_clients
        num_clients_test = len(test_ids) if num_clients is None else num_clients
        logger.info(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')

        # local dataset
        train_data_local_dict, train_data_local_num_dict = {}, {}
        test_data_local_dict, test_data_local_num_dict = {}, {}
        idx = 0

        for client_idx in range(num_clients_train):
            client_id = train_ids[client_idx]

            # train
            train_x = np.expand_dims(train_data['examples'][client_id]['pixels'][()], axis=1)
            train_y = train_data['examples'][client_id]['label'][()]

            digits_index = np.arange(len(train_y))[np.isin(train_y, range(10))]
            if client_idx < 2000:
                # client with only digits
                train_y = train_y[digits_index]
                train_x = train_x[digits_index]
            else:
                # client with only characters (but it's label noise for digits classification)
                non_digits_index = np.invert(np.isin(train_y, range(10)))
                train_y = train_y[non_digits_index]
                train_y = np.random.randint(10, size=len(train_y))
                train_x = train_x[non_digits_index]
            
            if len(train_y) == 0:
                continue
            
            # test
            test_x = np.expand_dims(test_data['examples'][client_id]['pixels'][()], axis=1)
            test_y = test_data['examples'][client_id]['label'][()]

            non_digits_index = np.invert(np.isin(test_y, range(10)))
            test_y[non_digits_index] = np.random.randint(10, size=sum(non_digits_index))
            
            if len(test_x) == 0:
                continue
                
            local_train_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
            train_data_local_dict[idx] = local_train_data
            train_data_local_num_dict[idx] = len(train_x)
                
            local_test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
            test_data_local_dict[idx] = local_test_data
            test_data_local_num_dict[idx] = len(test_x)

            idx += 1
        

        train_data.close()
        test_data.close()

    dataset = {}
    dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    return dataset