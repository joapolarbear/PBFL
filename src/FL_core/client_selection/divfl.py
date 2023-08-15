'''
Diverse Client Selection For Federated Learning via Submodular Maximization

Reference:
    https://openreview.net/pdf?id=nwKXyFvaUm
'''
from .client_selection import ClientSelection
import numpy as np
import math
import torch
from copy import deepcopy
from tqdm import tqdm
from itertools import product

from utils import logger

'''Diverse Client Selection'''
class DivFL(ClientSelection):
    def __init__(self, total, device, subset_ratio=0.1):
        super().__init__(total, device)
        '''
        Args:
            subset ratio: 0.1
        '''
        if subset_ratio is None:
            sys.exit('Please set the hyperparameter: subset ratio! =)')
        self.subset_ratio = subset_ratio

    def init(self, global_m, l=None):
        self.prev_global_m = global_m

    def select(self, n, client_idxs, metric, round=0, results=None):
        # pre-select
        '''
        ---
        Args
            metric: local_gradients
        '''
        # get clients' gradients
        local_grads = self.get_gradients(self.prev_global_m, metric)
        # get clients' dissimilarity matrix
        self.norm_diff = self.get_matrix_similarity_from_grads(local_grads).cpu().detach().numpy()
        # stochastic greedy
        selected_clients = self.lazy_greedy(len(client_idxs), n)
        return list(selected_clients)

    def get_gradients(self, global_m, local_models):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        local_model_params = []
        for model in local_models:
            local_model_params += [[tens.detach().cpu() for tens in list(model.parameters())]] #.numpy()

        global_model_params = [tens.detach().cpu() for tens in list(global_m.parameters())]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads += [[local_weights - global_weights
                                   for local_weights, global_weights in
                                   zip(local_params, global_model_params)]]

        return local_model_grads

    def get_matrix_similarity_from_grads(self, local_model_grads):
        """
        return the similarity matrix where the distance chosen to
        compare two clients is set with `distance_type`
        """
        n_clients = len(local_model_grads)
        metric_matrix = torch.zeros((n_clients, n_clients), device=self.device)
        for i, j in tqdm(product(range(n_clients), range(n_clients)), desc='>> similarity', total=n_clients**2, ncols=80):
            grad_1, grad_2 = local_model_grads[i], local_model_grads[j]
            for g_1, g_2 in zip(grad_1, grad_2):
                metric_matrix[i, j] += torch.sum(torch.square(g_1 - g_2))

        return metric_matrix

    def stochastic_greedy(self, num_total_clients, num_select_clients):
        # num_select_clients is the target number of selected clients each round,
        # subsample is a parameter for the stochastic greedy alg
        # initialize the ground set and the selected set
        V_set = set(range(num_total_clients))
        SUi = set()

        m = max(num_select_clients, int(self.subset_ratio * num_total_clients))
        for ni in range(num_select_clients):
            if m < len(V_set):
                R_set = np.random.choice(list(V_set), m, replace=False)
            else:
                R_set = list(V_set)
            if ni == 0:
                marg_util = self.norm_diff[:, R_set].sum(0)
                i = marg_util.argmin()
                client_min = self.norm_diff[:, R_set[i]]
            else:
                client_min_R = torch.minimum(client_min[:, None], self.norm_diff[:, R_set])
                marg_util = client_min_R.sum(0)
                i = marg_util.argmin()
                client_min = client_min_R[:, i]
            SUi.add(R_set[i])
            V_set.remove(R_set[i])
        return SUi
    
    def lazy_greedy(self,num_total_clients,  num_select_clients):
        # initialize the ground set and the selected set
        V_set = set(range(num_total_clients))
        SUi = set()

        S_util = 0
        marg_util = self.norm_diff.sum(0)
        i = marg_util.argmin()
        L_s0 = 2. * marg_util.max()
        marg_util = L_s0 - marg_util
        client_min = self.norm_diff[:,i]
        # print(i)
        SUi.add(i)
        V_set.remove(i)
        S_util = marg_util[i]
        marg_util[i] = -1.
        
        while len(SUi) < num_select_clients:
            argsort_V = np.argsort(marg_util)[len(SUi):]
            for ni in range(len(argsort_V)):
                i = argsort_V[-ni-1]
                SUi.add(i)
                client_min_i = np.minimum(client_min, self.norm_diff[:,i])
                SUi_util = L_s0 - client_min_i.sum()

                marg_util[i] = SUi_util - S_util
                if ni > 0:
                    if marg_util[i] < marg_util[pre_i]:
                        if ni == len(argsort_V) - 1 or marg_util[pre_i] >= marg_util[argsort_V[-ni-2]]:
                            S_util += marg_util[pre_i]
                            # print(pre_i, L_s0 - S_util)
                            SUi.remove(i)
                            SUi.add(pre_i)
                            V_set.remove(pre_i)
                            marg_util[pre_i] = -1.
                            client_min = client_min_pre_i.copy()
                            break
                        else:
                            SUi.remove(i)
                    else:
                        if ni == len(argsort_V) - 1 or marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                            S_util = SUi_util
                            # print(i, L_s0 - S_util)
                            V_set.remove(i)
                            marg_util[i] = -1.
                            client_min = client_min_i.copy()
                            break
                        else:
                            pre_i = i
                            SUi.remove(i)
                            client_min_pre_i = client_min_i.copy()
                else:
                    if marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                        S_util = SUi_util
                        # print(i, L_s0 - S_util)
                        V_set.remove(i)
                        marg_util[i] = -1.
                        client_min = client_min_i.copy()
                        break
                    else:
                        pre_i = i
                        SUi.remove(i)
                        client_min_pre_i = client_min_i.copy()
        return SUi
