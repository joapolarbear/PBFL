import numpy as np
import torch
import torch.nn as nn

from sklearn.mixture import GaussianMixture

from .client_selection import ClientSelection
from .fedcor_util import add_noise, lid_term, get_output

from utils import logger

class FedCor(ClientSelection):
    def __init__(self, args, total, device,
            seed = 13,
            correction = False,
            finetuning = False,
            finetune_frac = 0.05,
            finetune_iter_num = 200,
            # rounds2 = 200,
            relabel_ratio = 0.5,
            confidence_thres = 0.5,
            clean_set_thres = 0.1):
        super().__init__(total, device)
        
        self.args = args
        self.args.beta = 0
        
        self.seed = seed
        self.correction = correction
        self.finetuning = finetuning
        
        self.warmup_iter_num = args.warmup_iter_num
        self.warmup_frac = args.warmup_frac
        self.finetune_frac = finetune_frac
        self.finetune_iter_num = finetune_iter_num
        # self.rounds2 = rounds2
        self.relabel_ratio = relabel_ratio
        self.confidence_thres = confidence_thres
        self.clean_set_thres = clean_set_thres
        
        self.LID_accumulative_client = np.zeros(total)
        
        # 1: warmup phase, 2: finetuning phase (optional), 3: normal
        self.stage = 1
        self.stage_names = ["WARMUP", "FINETUNE", "NORMAL"]
        
        self.iter_cnt = 0
        self.iter_cnt_per_stage = 0
        self.sub_iter_num = 0
    
    @property
    def warmup_iter_end(self):
        return self.stage == 1 and self.sub_iter_num == int(1/self.warmup_frac)

    @property
    def warmup_end(self):
        return self.stage == 1 and self.iter_cnt >= self.warmup_iter_num

    @property
    def finetune_end(self):
        assert self.finetuning
        return self.stage == 2 and self.iter_cnt >= (self.warmup_iter_num + self.finetune_iter_num)
    
    # @property
    # def total_round_num(self):
    #     _total = self.warmup_iter_num * int(1/self.warmup_frac) + self.rounds2
    #     if self.finetuning:
    #         _total += self.finetune_iter_num
    #     return _total

    @property
    def stage_name(self):
        return self.stage_names[self.stage-1]
    
    def setup(self, train_sizes):
        self.train_sizes = train_sizes

    def init(self, global_model):
        logger.info(f">> [{self.stage_name} iter {self.iter_cnt_per_stage}/{self.iter_cnt}] init ... ")
        need_init = True
        if self.stage == 1:
            logger.info(f"   sub_iter_num = {self.sub_iter_num} ")
            if self.sub_iter_num == 0:
                need_init = True
            else:
                need_init = False
        else:
            need_init = True
        
        if not need_init:
            return
        
        self.loss_whole = dict([(client_id, np.zeros(
            self.train_sizes[client_id])) for client_id in range(self.total)])
        self.loss_accumulative_whole = dict([(client_id, np.zeros(
            self.train_sizes[client_id])) for client_id in range(self.total)])
        self.LID_client = np.zeros(self.total)
        self.prob = [1 / self.total] * self.total

        if self.stage == 1:
            if self.iter_cnt == 0:
                self.mu_list = np.zeros(self.total)
            else:
                self.mu_list = self.estimated_noisy_level
    
    def get_mu(self, client_id):
        if self.stage == 1:
            return self.mu_list[client_id]
        else:
            return 0
        
    def select(self):
        if self.stage == 1:
            selected_client_idxs = np.random.choice(
                range(self.total), int(self.total*self.warmup_frac), p=self.prob, replace=False)
            self.sub_iter_num += 1
        elif self.stage == 2 or self.stage == 3:
            selected_client_idxs = np.random.choice(
                range(self.total), self.m, replace=False, p=self.prob)
            self.iter_cnt += 1
            self.iter_cnt_per_stage += 1
        else:
            raise
        
        return selected_client_idxs

    def fix_prob(self):
        ### Make the sum of self.prob to 1
        self.prob = [self.prob[i] / sum(self.prob) for i in range(len(self.prob))]

    def warmup_sub_iter_summary(self, client_id, client_output_array, client_loss_array):
        LID_local = list(lid_term(client_output_array, client_output_array))
        self.loss_whole[client_id] = client_loss_array
        self.LID_client[client_id] = np.mean(LID_local)
        
        self.prob[client_id] = 0
        if sum(self.prob) > 0:
            self.fix_prob()
        
    def warmup_iter_summary(self):
        self.LID_accumulative_client = self.LID_accumulative_client + np.array(self.LID_client)
        for client_id in self.loss_accumulative_whole:
            self.loss_accumulative_whole[client_id] += + np.array(self.loss_whole[client_id])

        # Apply Gaussian Mixture Model to LID
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=self.seed).fit(
            np.array(self.LID_accumulative_client).reshape(-1, 1))
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(self.LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        self.noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]

        self.estimated_noisy_level = np.zeros(self.total)

        for client_id in self.noisy_set:
            client_loss_array = np.array(self.loss_accumulative_whole[client_id])   
            gmm_loss = GaussianMixture(n_components=2, random_state=self.seed).fit(np.array(client_loss_array).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(client_loss_array).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            self.estimated_noisy_level[client_id] = len(pred_n) / self.train_sizes[client_id]
        
        self.sub_iter_num = 0
        self.iter_cnt += 1
        self.iter_cnt_per_stage += 1

        if self.warmup_end:
            self.end_warmup()
    
    def end_warmup(self):
        self.iter_cnt_per_stage = 0
        self.args.beta = 0
        if self.finetuning:
            self.stage = 2
            selected_clean_idx = np.where(self.estimated_noisy_level <= self.clean_set_thres)[0]
            self.prob = np.zeros(self.total) # np.zeros(100)
            self.prob[selected_clean_idx] = 1 / len(selected_clean_idx)
            self.fix_prob()
            self.m = max(int(self.finetune_frac * self.total), 1)  # num_select_clients
            self.m = min(self.m, len(selected_clean_idx))
        else:
            self.stage = 3
            self.m = max(int(self.finetune_frac * self.total), 1)  # num_select_clients
            self.prob = [1/self.total for i in range(self.total)]

    def correct_dataset(self, idx, client_output_array, client_loss_array):
        relabel_idx = (-client_loss_array).argsort()[:int(self.train_sizes[idx] * self.estimated_noisy_level[idx] * self.relabel_ratio)]
        relabel_idx = list(set(np.where(np.max(client_output_array, axis=1) > self.confidence_thres)[0]) & set(relabel_idx))
        
        # ### TODO (huhanpeng): why do we need to update the training dataset
        raise NotImplementedError()
        # y_train_noisy_new = np.array(dataset_train.targets)
        # y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(client_output_array, axis=1)[relabel_idx]
        # dataset_train.targets = y_train_noisy_new
    
    def end_finetune(self):
        self.iter_cnt_per_stage = 0
        self.stage = 3
        self.m = max(int(self.finetune_frac * self.total), 1)  # num_select_clients
        self.prob = [1/self.total for i in range(self.total)]
        