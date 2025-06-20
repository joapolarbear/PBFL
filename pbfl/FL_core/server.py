import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp
import random
import copy

from .client import Client
from .client_selection.config import *
from .trainer import Trainer
from ..utils import logger

from torch.utils.data import TensorDataset

def print_selected_client(client_indices, THRESHOLD = 100):
    rst = sorted([str(i) for i in client_indices])
    if len(rst) <= THRESHOLD:
        logger.info(f'Selected clients: [{", ".join(rst)}]')
    else:
        logger.info(f'Selected clients: [{", ".join(rst[:THRESHOLD])} ... ]')


class Server(object):
    def __init__(self, data, init_model, args, selection, fed_algo, files):
        """
        Server to execute
        ---
        Args
            data: dataset for FL
            init_model: initial global model
            args: arguments for overall FL training
            selection: client selection method
            fed_algo: FL algorithm for aggregation at server
            results: results for recording
        """
        self.train_data = data['train']['data']
        self.train_sizes: dict[int, int] = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes: dict[int, int] = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()

        self.device = args.device
        self.args = args
        self.global_model = init_model
        self.selection_method = selection
        self.selection_method.server = self
        self.federated_method = fed_algo
        self.files = files

        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.num_available = args.num_available
        if self.num_available is not None:
            random.seed(args.seed)

        self.total_round = args.num_round
        self.save_results = not args.no_save_results
        self.save_probs = args.save_probs

        if self.save_probs:
            num_local_data = np.array([self.train_sizes[idx] for idx in range(args.total_num_client)])
            num_local_data.tofile(files['num_samples'], sep=',')
            files['num_samples'].close()
            del files['num_samples']

        self.test_on_training_data = False

        ## INITIALIZE
        # initialize the training status of each client
        self._init_clients(init_model)

        # initialize the client selection method
        if self.args.method in NEED_BEFORE_TRAIN_METHOD:
            self.selection_method.before_train(self.train_sizes, self.global_model)

        if self.args.method in LOSS_THRESHOLD:
            self.ltr = 0.0

        self.global_trainer = Trainer(self.args)
        all_data_by_label = {}
        for client_id, local_data in self.test_data.items():
            batch_x, batch_y = local_data.tensors
            for x, y in zip(batch_x, batch_y):
                label = int(y)
                if label not in all_data_by_label:
                    all_data_by_label[label] = [[], []]
                all_data_by_label[label][0].append(x)
                all_data_by_label[label][1].append(y)
        min_sample = min([len(data_pair[1]) for data_pair in all_data_by_label.values()])
        xss = []
        yss = []
        for label, data_pair in all_data_by_label.items():
            xs, ys = data_pair
            selected_data_idx = np.random.choice(
                len(ys), min_sample, replace=False)
            xss.append(torch.stack(xs)[selected_data_idx])
            yss.append(torch.stack(ys)[selected_data_idx])

        X = torch.cat(xss, dim=0)
        Y = torch.cat(yss, dim=0)
        perm = torch.randperm(len(Y))
        self.global_test_data = TensorDataset(X[perm], Y[perm])
        logger.info(f"Global test data size: {len(Y)}")
   
    def _init_clients(self, init_model):
        """
        initialize clients' model
        ---
        Args
            init_model: initial given global model
        """
        self.client_list = []
        for client_idx in range(self.total_num_client):
            local_train_data = self.train_data[client_idx]
            local_test_data = self.test_data[client_idx] if client_idx in self.test_clients else np.array([])
            c = Client(client_idx, self.train_sizes[client_idx], local_train_data,
                local_test_data, self.args)
            self.client_list.append(c)

    def global_test(self):
        if self.global_trainer is None:
            return {}
        result = self.global_trainer.test(self.global_model, self.global_test_data)
        return result

    def train(self):
        """
        FL training
        """
        ## ITER COMMUNICATION ROUND
        for round_idx in range(self.total_round):
            print()
            logger.info(f'ROUND {round_idx}')

            ## GET GLOBAL MODEL
            #self.global_model = self.trainer.get_model()
            self.global_model = self.global_model.to(self.device)

            if self.args.dataset=='cifar' or round_idx in self.args.schedule:
                self.args.lr_local *= self.args.lr_decay
            
            ##################################################################
            #                        Set clients
            ##################################################################
            client_indices = [*range(self.total_num_client)]
            if self.num_available is not None:
                logger.info(f'available clients {self.num_available}/{len(client_indices)}')
                np.random.seed(self.args.seed + round_idx)
                client_indices = np.random.choice(client_indices, self.num_available, replace=False)
                self.save_selected_clients(round_idx, client_indices)

            ##################################################################
            #                        Set client selection methods
            ##################################################################
            # initialize selection methods by setting given global model
            if self.args.method in NEED_BEFORE_STEP_METHOD:
                if self.args.method in [
                    SelectMethod.gpfl, SelectMethod.divfl, 
                    "FedCorr", SelectMethod.cosin,
                    SelectMethod.hisc,
                ]:
                    self.selection_method.before_step(self.global_model)
                else:
                    raise NotImplementedError("We do not maintain a cost model for each client, "
                        "so we fail to get the local model before client selection. \n"
                        "\t Methods requiring init only include `Cluster2`")
                    local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                    self.selection_method.before_step(self.global_model, local_models)
                    del local_models
            # candidate client selection before local training
            if self.args.method in CANDIDATE_SELECTION_METHOD:
                logger.info(f'Candidate client selection {self.args.num_candidates}/{len(client_indices)}')
                client_indices = self.selection_method.select_candidates(client_indices, self.args.num_candidates)
                print_selected_client(client_indices)

            ##################################################################
            #                        PRE-CLIENT SELECTION
            ##################################################################
            # client selection before local training (for efficiency)
            if self.args.method in PRE_SELECTION_METHOD:
                if self.args.method in [SelectMethod.gpfl, SelectMethod.cosin, SelectMethod.hisc]:
                    num_before = len(client_indices)
                    client_indices = self.selection_method.select(
                        self.num_clients_per_round, 
                        client_idxs=client_indices,
                        round=round_idx,
                        metric=None
                    )
                    logger.info(f'Pre-client selection {num_before} -> {len(client_indices)}')
                elif self.args.method == "FedCorr":
                    num_before = len(client_indices)
                    client_indices = self.selection_method.select()
                    logger.info(f'Pre-client selection {num_before} -> {len(client_indices)}')
                else:
                    num_before = len(client_indices)
                    client_indices = self.selection_method.select(self.num_clients_per_round, client_indices, None)
                    logger.info(f'Pre-client selection {num_before} -> {len(client_indices)}')
                print_selected_client(client_indices)

            ##################################################################
            #                        CLIENT UPDATE (TRAINING)
            ##################################################################
            engaged_client_indices = deepcopy(client_indices)
            ### TODO huhanpeng: add a L2(M_local-M_global) to the loss function
            local_losses, accuracy, local_metrics = self.train_clients(client_indices)

            ##################################################################
            #                        POST-CLIENT SELECTION
            ##################################################################
            if self.args.method not in PRE_SELECTION_METHOD:
                logger.info(f'Post-client selection {self.num_clients_per_round}/{len(client_indices)}')
                # select by local models(gradients)
                if self.args.method in NEED_LOCAL_MODELS_METHOD:
                    local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                    selected_client_indices = self.selection_method.select(
                        self.num_clients_per_round,
                        client_idxs=client_indices,
                        round=round_idx,
                        results=self.files['prob'] if self.save_probs else None, metric=local_models
                    )
                    del local_models
                # select by local losses
                else:
                    selected_client_indices = self.selection_method.select(
                        self.num_clients_per_round,
                        client_idxs=client_indices,
                        round=round_idx,
                        results=self.files['prob'] if self.save_probs else None, metric=local_metrics
                    )
                if self.args.method in CLIENT_UPDATE_METHOD:
                    for idx in client_indices:
                        self.client_list[idx].update_ema_variables(round_idx)
                # update local metrics
                client_indices = np.take(client_indices, selected_client_indices).tolist()
                print_selected_client(client_indices)
                local_losses = np.take(local_losses, selected_client_indices)
                accuracy = np.take(accuracy, selected_client_indices)

            ## CHECK and SAVE current updates
            # self.weight_variance(local_models) # check variance of client weights
            self.save_current_updates(local_losses, accuracy, len(client_indices), phase='Train', round=round_idx)
            self.save_selected_clients(round_idx, client_indices)
            
            # DEBUGGING
            if self.args.method not in [
                SelectMethod.gpfl, "FedCorr", SelectMethod.cosin
            ]:
                assert len(client_indices) == self.num_clients_per_round, \
                    (len(client_indices), self.num_clients_per_round)
                    
            ##################################################################
            #                        SERVER AGGREGATION
            ##################################################################
            self.aggregate_model(client_indices)
            
            ##################################################################
            #                        POST-process for each selection method
            ##################################################################
            self.selection_method.post_process(engaged_client_indices)
            
            ##################################################################
            #                        TEST
            ##################################################################  
            self.global_model.eval()
            if self.test_on_training_data:
                # test on train dataset
                raise ValueError("Why should we test on the training data")
                self.test(self.total_num_client, phase='TrainALL')
                self.test_on_training_data = False
            # test on test dataset
            result: dict = self.global_test()
            local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
            if self.args.method in NEED_AFTER_STEP_METHOD:
                self.selection_method.after_step(client_indices, local_models, self.global_model, result["loss"], result["acc"])
            # self.test(len(self.test_clients), phase='Test')
            phase='Test'
            self.record[f'{phase}/Loss'] = result["loss"]
            self.record[f'{phase}/Acc'] = result["acc"]

            ##################################################################
            #                        Record log info 
            ##################################################################  
            logger.info('[ROUND {}] {}ing: Loss {:.6f} Acc {:.4f}'.format(round_idx, phase, result["loss"], result["acc"]))

            if self.args.wandb:
                wandb.log(self.record)

            ## Clear garbages
            del local_models, local_losses, accuracy
            for client_idx in engaged_client_indices:
                self.client_list[client_idx].trainer.clear_model()

        for k in self.files:
            if self.files[k] is not None:
                self.files[k].close()

    def aggregate_model(self, selected_client_idxs):
        # aggregate local models
        local_models = [self.client_list[idx].trainer.get_model() for idx in selected_client_idxs]
        if self.args.fed_algo == 'FedAvg':
            global_model_params = self.federated_method.update(local_models, selected_client_idxs)
        else:
            global_model_params = self.federated_method.update(
                local_models, selected_client_idxs, self.global_model, self.client_list)
        
        # update aggregated model to global model
        self.global_model.load_state_dict(global_model_params)
        del local_models
            
    def local_training(self, client_idx):
        """
        train one client with the global model
        ---
        Args
            client_idx: client index for training
        Return
            result: trained model, (total) loss value, accuracy
        """
        client = self.client_list[client_idx]
        if self.args.method in LOSS_THRESHOLD:
            client.trainer.update_ltr(self.ltr)
        
        if self.args.method == "FedCorr":
            mu = self.selection_method.get_mu(client_idx)
            result = client.train(self.global_model, mu=mu)
        else:
            result = client.train(self.global_model)
        return result

    def local_testing(self, client_idx, use_local_model=False):
        """
        test one client with the global model
        ---
        Args
            client_idx: client index for test
            results: loss, acc, auc
        """
        client = self.client_list[client_idx]
        result = client.test(self.global_model, self.test_on_training_data,
                            use_local_model=use_local_model)
        return result

    def train_clients(self, client_indices):
        """
        train multiple clients (w. or w.o. multi processing) with 
        the global model
        ---
        Args
            client_indices: client indices for training
        Return
            trained models, loss values, accuracies
        """
        local_losses, accuracy, local_metrics = [], [], []
        ll, lh = np.inf, 0.
        # local training with multi processing
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = list(pool.imap(self.local_training, client_indices))

                result = {k: [result[idx][k] for idx in range(len(result))] for k in result[0].keys()}
                local_losses.extend(result['loss'])
                accuracy.extend(result['acc'])
                local_metrics.extend(result['metric'])
                
                if self.args.method in LOSS_THRESHOLD:
                    if min(result['llow']) < ll: ll = min(result['llow'])
                    lh += sum(result['lhigh'])
        # local training without multi processing
        else:
            for client_idx in client_indices:
                result = self.local_training(client_idx)

                local_losses.append(result['loss'])
                accuracy.append(result['acc'])
                local_metrics.append(result['metric'])

                if self.args.method in LOSS_THRESHOLD:
                    if result['llow'] < ll: ll = result['llow'].item()
                    lh += result['lhigh']

        if self.args.method in LOSS_THRESHOLD:
            lh /= len(client_indices)
            self.ltr = self.selection_method.update(lh, ll, self.ltr)

        return local_losses, accuracy, local_metrics

    def test(self, num_clients_for_test, phase='Test', save=True, 
             use_local_model=False):
        """
        test multiple clients on their respective dataset
        ---
        Args
            num_clients_for_test: int or numpy or list
                number of clients for test
                
            TrainALL: test on train dataset
            Test: test on test dataset
        """
        
        if isinstance(num_clients_for_test, int):
            ### TODO (huhanpeng): remove this assertation
            assert num_clients_for_test == self.total_num_client
            clients_to_test = list(range(num_clients_for_test))
        else:
            assert save is False, ("Do not support to save test"
                "results with partial of the clients")
            clients_to_test = list(num_clients_for_test)
        
        metrics = {'loss': [], 'acc': []}
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = list(tqdm(pool.imap(self.local_testing, [*clients_to_test], use_local_model),
                                   desc=f'>> local testing on {phase} set'))
                result = {k: [result[idx][k] for idx in range(len(result))] for k in result[0].keys()}
                metrics['loss'].extend(result['loss'])
                metrics['acc'].extend(result['acc'])
        else:
            for client_idx in clients_to_test:
                result = self.local_testing(client_idx, 
                                    use_local_model=use_local_model)
                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])
        if save:
            self.save_current_updates(metrics['loss'], metrics['acc'], num_clients_for_test, phase=phase)
        return metrics

    def try_federated_learning(self, client_indices):
        ''' Try to train selected clients without affect the 
        global model.
            return the test resutls, including the average 
            accuracy and the list of loss on each client's test data
        '''
        _global_model = copy.deepcopy(self.global_model)
        self.train_clients(client_indices)
        self.aggregate_model(client_indices)
        # metrics = self.test(client_indices, phase='Test', 
        #                 save=False, use_local_model=False)
        self.global_model = _global_model
        # return metrics
        
    def save_current_updates(self, losses, accs, num_clients, phase='Train', round=None):
        """
        update current updated results for recording
        ---
        Args
            losses: losses
            accs: accuracies
            num_clients: number of clients
            phase: current phase (Train or TrainALL or Test)
            round: current round
        Return
            record "Round,TrainLoss,TrainAcc,TestLoss,TestAcc"
        """
        loss, acc = sum(losses) / num_clients, sum(accs) / num_clients

        if phase == 'Train':
            self.record = {}
            self.round = round
        self.record[f'{phase}/Loss'] = loss
        self.record[f'{phase}/Acc'] = acc
        status = num_clients if phase == 'Train' else 'ALL'

        logger.info('{} Clients {}ing: Loss {:.6f} Acc {:.4f}'.format(status, phase, loss, acc))

        if phase == 'Test':
            if self.args.wandb:
                wandb.log(self.record)
            if self.save_results:
                if self.test_on_training_data:
                    tmp = '{:.8f},{:.4f},'.format(self.record['TrainALL/Loss'], self.record['TrainALL/Acc'])
                else:
                    tmp = ''
                rec = '{},{:.8f},{:.4f},{}{:.8f},{:.4f}\n'.format(self.round,
                                                                  self.record['Train/Loss'], self.record['Train/Acc'], tmp,
                                                                  self.record['Test/Loss'], self.record['Test/Acc'])
                self.files['result'].write(rec)

    def save_selected_clients(self, round_idx, client_indices):
        """
        save selected clients' indices
        ---
        Args
            round_idx: current round
            client_indices: clients' indices to save
        """
        self.files['client'].write(f'{round_idx+1},')
        np.array(client_indices).astype(int).tofile(self.files['client'], sep=',')
        self.files['client'].write('\n')

    def weight_variance(self, local_models):
        """
        calculate the variances of model weights
        ---
        Args
            local_models: local clients' models
        """
        variance = 0
        for k in tqdm(local_models[0].state_dict().keys(), desc='>> compute weight variance'):
            tmp = []
            for local_model_param in local_models:
                tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
            variance += torch.var(torch.tensor(tmp), dim=0)
        variance /= len(local_models)
        logger.info('variance of model weights {:.8f}'.format(variance))
