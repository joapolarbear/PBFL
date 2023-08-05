import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import json

from .base_dataset import BaseDataset


class FederatedEMNISTDatasetIID(BaseDataset):
    def __init__(self, data_dir, args):
        super(FederatedEMNISTDatasetIID, self).__init__()
        self.num_classes = 62
        # self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        # self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.min_num_samples = 10
        # min_num_samples = 150; num_clients = 2492
        # min_num_samples = 100; num_clients = 

        self._init_data(data_dir)
        self.train_num_clients = len(self.dataset['train']['data_sizes'].keys())
        self.test_num_clients = len(self.dataset['test']['data_sizes'].keys())
        print(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_IID.pickle')
        if os.path.isfile(file_name):
            print('> read dataset ...')
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            print('> create dataset ...')
            dataset = preprocess(data_dir, self.min_num_samples)
            # with open(file_name, 'wb') as f:
            #     pickle.dump(dataset, f)
        self.dataset = dataset

def preprocess(data_dir, min_num_samples):
    # local dataset we return
    ''' train_data_local_dict: {client_id: TensorDataset(x, y)}
        train_data_local_num_dict: {client_id: dataset_size}
    '''
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
        
        ''' data in one h5 file
            {
                "examples": {
                    # user_names: {'pixels': x, 'label': y}
                    'f4031_33': ...,
                    'f4085_42': ...
                },
                ...
            }
        '''
        train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
        test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')
        
        train_ids = list(train_data['examples'].keys())

        test_ids = list(test_data['examples'].keys())
        num_clients_train = len(train_ids)
        num_clients_test = len(test_ids)
        print(f'#TrainClients {num_clients_train} #TestClients {num_clients_test}')

        idx = 0

        for client_idx in range(num_clients_train):
            client_id = train_ids[client_idx]

            # train
            train_x = np.expand_dims(train_data['examples'][client_id]['pixels'][()], axis=1)
            train_y = train_data['examples'][client_id]['label'][()]

            if len(train_x) < min_num_samples:
                continue
            train_x = train_x[:min_num_samples]
            train_y = train_y[:min_num_samples]

            local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
            train_data_local_dict[idx] = local_data
            train_data_local_num_dict[idx] = len(train_x)

            # test
            test_x = np.expand_dims(test_data['examples'][client_id]['pixels'][()], axis=1)
            test_y = test_data['examples'][client_id]['label'][()]
            local_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
            test_data_local_dict[idx] = local_data
            test_data_local_num_dict[idx] = len(test_x)
            if len(test_x) == 0:
                print(client_idx)
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

# import json
# import numpy as np
# import h5py
# import os

# with open("mytest.json", 'r') as fp:
#     test_data = json.load(fp)

# X1 = np.empty((0, 784), dtype=float)
# for xypair in test_data['user_data'].values():
#     x = np.array(xypair["x"])
#     X1 = np.concatenate((X1, x), axis=0)
#     y = xypair["y"]

# np.std(X1, axis=0)


# data_dir = "."
# test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')

# X2 = np.empty((0, 784), dtype=float)
# for xypair in test_data['examples'].values():
#     x = np.array(xypair["pixels"][()]).reshape(-1, 784)
#     X2 = np.concatenate((X2, x), axis=0)
#     if len(X2) > len(X1):
#         break

# np.std(X2, axis=0)

# rst = np.nan_to_num(np.std(X1, axis=0) / np.std(X2, axis=0))
# for p in [5, 10, 25, 50, 75, 90, 10]:   
#     print(p, ": ", np.percentile(rst, p))
