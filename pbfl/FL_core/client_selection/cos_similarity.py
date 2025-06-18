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

from .gpfl import Proj_Bandit

class CosineSimilaritySelector(Proj_Bandit):  
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
                cosin_similarity = self.compute_cosine_similarity(
                    local_grad_per_key, global_grad_per_key)
                proj_list.append(cosin_similarity)
            idxs_proj.append(torch.mean(torch.Tensor(proj_list)))

        # import pdb; pdb.set_trace()
        final_reward = torch.nn.Softmax(dim=0)(torch.Tensor(idxs_proj)) * improved
        # print("projection after softmax", final_reward)
        for client_idx, reward in zip(selected_client_idxs, (final_reward)):
            self.client2rewards[client_idx].append((reward))

        for client_idx in range(len(self.client2proj)):
            self.client2proj[client_idx] = np.mean(self.client2rewards[client_idx])

    def compute_cosine_similarity(self, local_grad, global_grad):
        # Compute cosine similarity between local and global gradients
        return torch.nn.functional.cosine_similarity(local_grad, global_grad, dim=0)