from .client_selection import ClientSelection
import numpy as np
import math
import torch
from copy import deepcopy
from tqdm import tqdm
from itertools import product
import sys
import signal
import atexit

from utils import logger

class Proj_Bandit(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

        self.client2rewards = []
        for _ in range(total):
            self.client2rewards.append([0])
        self.client2proj = np.array([-math.inf] * total)

        self.client2selected_cnt = np.array([0] * total)
        self.client_update_cnt = 0
        self.total = total

        self.global_accu = 0
        self.global_loss = 1e6
        
        self.warmup_bound = 20
        self.warmup_frac = 1 / self.warmup_bound
        
        # signal.signal(signal.SIGINT, self.signal_handler)
        # signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.signal_handler)
    
    def signal_handler(self, *args):
        print()
        print(self.client2selected_cnt)
        sys.exit(0)
    
    def setup(self, n_samples):
        self.accuracy_per_update = [self.global_accu]
        self.loss_per_update = [self.global_loss]

    def init(self, global_m, l=None):
        self.prev_global_params = deepcopy([tens.detach().to(self.device) for tens in list(global_m.parameters())])

    def update_proj_list(self, selected_client_idxs, global_m, local_models, improved):
        """
        return the `projected gradient` 
        """

        local_model_params = []
        for model in local_models:
            local_model_params.append([tens.detach().to(self.device) for tens in list(model.parameters())]) #.numpy()
        
        prev_global_model_params = self.prev_global_params
        global_model_params = [tens.detach().to(self.device) for tens in list(global_m.parameters())]
        global_grad = [(global_weights - prev_global_weights).flatten()
                                   for global_weights, prev_global_weights in
                                   zip(global_model_params, prev_global_model_params)]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads.append([(local_weights - prev_global_weights).flatten()
                                   for local_weights, prev_global_weights in
                                   zip(local_params, prev_global_model_params)])

        idxs_proj = []
        grad_norm = [torch.sqrt(torch.sum(global_grad_per_key**2)) for global_grad_per_key in global_grad]
        for local_grad in local_model_grads:
            proj_list = []
            for local_grad_per_key, global_grad_per_key, grad_norm_per_key in zip(local_grad, global_grad, grad_norm):
                proj_list.append(torch.dot(local_grad_per_key, global_grad_per_key) / grad_norm_per_key)
            idxs_proj.append(torch.mean(torch.Tensor(proj_list)))

        # import pdb; pdb.set_trace()
        final_reward = torch.nn.Softmax(dim=0)(torch.Tensor(idxs_proj)) * improved
        # print("projection after softmax", final_reward)
        for client_idx, reward in zip(selected_client_idxs, (final_reward)):
            self.client2rewards[client_idx].append((reward))

        for client_idx in range(len(self.client2proj)):
            self.client2proj[client_idx] = np.mean(self.client2rewards[client_idx])
    
    def get_ucb(self, a):
        momemtum_based_grad_proj = self.client2proj
        # print("Proj", momemtum_based_grad_proj)
        assert isinstance(self.client2proj, list) or isinstance(self.client2proj, np.ndarray)
        # assert len(self.client2proj) == num_of_client

        alpha = a/800
        
        ucb = self.client2proj + alpha * np.sqrt(
            (2 * np.log(self.client_update_cnt))/self.client2selected_cnt)
        
        # print("ucb", ucb)
        return ucb
    
    def post_update(self, client_idxs, local_models, global_m):
        # import pdb; pdb.set_trace()
        self.accuracy_per_update.append(self.global_accu)
        self.loss_per_update.append(self.global_loss)

        if self.accuracy_per_update[-1] > self.accuracy_per_update[-2]:
            ### Better accuracy, larger projection is better
            improved = 1
        elif self.accuracy_per_update[-1] == self.accuracy_per_update[-2]:
            if self.loss_per_update[-1] < self.loss_per_update[-2]:
                improved = 0.5
            elif self.loss_per_update[-1] > self.loss_per_update[-2]:
                improved = -0.5
            else: 
                improved = 0
        else:
            ### Worse accuracy, smaller projection is better
            improved = -1

        self.update_proj_list(client_idxs, global_m, local_models, improved)

    def select(self, n, client_idxs, metric, round=0, results=None):
        # pre-select
        '''
        ---
        Args
            metric: local_gradients
        '''
        
        
        if self.client_update_cnt < self.warmup_bound:
            MAX_SELECTED_NUM = math.ceil(self.total * self.warmup_frac)
            logger.info(f"> GPFL warmup {self.client_update_cnt}")
            selected_client_index = np.random.choice(
                self.total, MAX_SELECTED_NUM, replace=False)
            # st = self.client_update_cnt * MAX_SELECTED_NUM
            # ed = st + MAX_SELECTED_NUM
            # ed = min(ed, self.total)
            # selected_client_index = np.arange(st, ed)
            # selected_client_index = np.arange(self.total)
            # selected_client_index = np.random.choice(self.total, n, replace=False)
        else:
            ucb = self.get_ucb(a=self.client_update_cnt)
            sorted_client_idxs = ucb.argsort()[::-1]
            ### Select clients
            selected_client_index = sorted_client_idxs[:n]
            
        for client_idx in selected_client_index:
            self.client2selected_cnt[client_idx] += 1
        self.client_update_cnt += 1

        return selected_client_index.astype(int)