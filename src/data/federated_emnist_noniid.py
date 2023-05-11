import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import json


class FederatedEMNISTDataset_nonIID:
    def __init__(self, data_dir, args):
        '''
        known class: digits (10)
        unknown class: characters (52) -> label noise
        '''
        self.num_classes = 10
        self.min_num_samples = 10
        self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')
        self.train_num_clients = len(self.dataset['train']['data_sizes'].keys())
        self.test_num_clients = len(self.dataset['test']['data_sizes'].keys())
        print(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')
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


def preprocess(data_dir, min_num_samples):
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    if True:
        # Load the JSON file
        with open('../dataset/FederatedEMNIST/mytrain.json', 'r') as f:
            train_data = json.load(f)
        # Convert the data to NumPy arrays
        num_samples = np.array(train_data['num_samples']) # 
        user_names = np.array(train_data['users']) # user_names corresponding to the array of num_samples
        user_data = train_data['user_data'] # {user_name: {'x': x, 'y': y}}

        with open('../dataset/FederatedEMNIST/mytest.json', 'r') as f:
            test_data = json.load(f)
        # Convert the data to NumPy arrays
        num_samples = np.array(test_data['num_samples']) # 
        user_names = np.array(test_data['users']) # user_names corresponding to the array of num_samples
        user_data = test_data['user_data'] # {user_name: {'x': x, 'y': y}}

        train_ids = list(train_data['user_data'].keys())

        test_ids = list(test_data['user_data'].keys())
        num_clients_train = len(train_ids)
        num_clients_test = len(test_ids)
        print(f'#TrainClients {num_clients_train} #TestClients {num_clients_test}')

        idx = 0
        for client_idx in range(num_clients_train):
            client_id = train_ids[client_idx]


            # train
            train_x = np.expand_dims(train_data['user_data'][client_id]['x'], axis=1)
            train_y = train_data['user_data'][client_id]['y']

            if len(train_x) < min_num_samples:
                continue
            train_x = train_x[:min_num_samples]
            train_y = train_y[:min_num_samples]

            dim1, dim2, _ = train_x.shape
            train_x = train_x.reshape(dim1, dim2, 28, 28)

            local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
            train_data_local_dict[idx] = local_data
            train_data_local_num_dict[idx] = len(train_x)

            # test
            test_x = np.expand_dims(test_data['user_data'][client_id]['x'], axis=1)
            dim1, dim2, _ = test_x.shape
            test_x = test_x.reshape(dim1, dim2, 28, 28)
            test_y = test_data['user_data'][client_id]['y']
            local_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
            test_data_local_dict[idx] = local_data
            test_data_local_num_dict[idx] = len(test_x)
            if len(test_x) == 0:
                print(client_idx)
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
        print(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')

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