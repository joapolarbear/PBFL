'''
Client Selection for Federated Learning
'''
import os
import time

AVAILABLE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    AVAILABLE_WANDB = False
    
import torch
import random

import utils
args = utils.init(".", "pbfl")
from utils import logger

from data import load_data
from model import create_model
from FL_core.server import Server
from FL_core.client_selection import *
from FL_core.federated_algorithm import *


def federated_algorithm(dataset, model, args):
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        return FedAvg(train_sizes, model)


def client_selection_method(args):
    #total = args.total_num_client if args.num_available is None else args.num_available
    kwargs = {'total': args.total_num_client, 'device': args.device}
    if args.method == SelectMethod.random:
        return RandomSelection(**kwargs)
    elif args.method == SelectMethod.afl:
        return ActiveFederatedLearning(**kwargs, args=args)
    elif args.method == SelectMethod.cluster1:
        return ClusteredSampling1(**kwargs, n_cluster=args.num_clients_per_round)
    elif args.method == SelectMethod.cluster2:
        return ClusteredSampling2(**kwargs, dist=args.distance_type)
    elif args.method == SelectMethod.pow_d:
        assert args.num_candidates is not None
        return PowerOfChoice(**kwargs, d=args.num_candidates)
    elif args.method == SelectMethod.divfl:
        assert args.subset_ratio is not None
        return DivFL(**kwargs, subset_ratio=args.subset_ratio)
    elif args.method == 'GradNorm':
        return GradNorm(**kwargs)
    elif args.method == SelectMethod.gpfl:
        return Proj_Bandit(args, **kwargs)
    elif args.method == SelectMethod.fedcor:
        return FedCor(args, **kwargs)
    elif args.method == SelectMethod.single:
        args.total_num_client = args.num_clients_per_round = 1
        return SingleSelection(**kwargs)
    elif args.method == SelectMethod.cosin:
        return CosineSimilaritySelector(args, **kwargs)
    elif args.method == SelectMethod.hisc:
        return HiCSSelector(args, **kwargs)
    else:
        raise('CHECK THE NAME OF YOUR SELECTION METHOD')



if __name__ == '__main__':
    args.start = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if args.comment:
        args.comment = f"-{args.comment}"
    #if args.labeled_ratio < 1:
    #    args.comment = f"-L{args.labeled_ratio}{args.comment}"
    if args.fed_algo != 'FedAvg':
        args.comment = f"-{args.fed_algo}{args.comment}"
    
    # save to wandb
    args.wandb = AVAILABLE_WANDB
    exp_name_short = os.getenv('EXP_NAME_SHORT')
    assert exp_name_short is not None
    if args.wandb:
        exp_name = os.environ.get("PBFL_EXP_NAME", None)
        if exp_name:
            wandb_dir = f"{exp_name}-wandb"
            os.makedirs(wandb_dir, exist_ok=True)
        else:
            wandb_dir = '../'
        wandb.init(
            project=f'PBFL-{args.dataset}',
            name=f"{args.start}-{exp_name_short}",
            config=args,
            dir=wandb_dir,
            save_code=True,
            mode='online'
        )
        # wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or 'main.py' in x)

    # fix seed
    if True or args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # device setting
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        if ',' in args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        args.device = torch.device(f"cuda:{args.gpu_id[0]}")
        torch.cuda.set_device(args.device)
        logger.info(f'Current cuda device {torch.cuda.current_device()}')

    # set data
    data = load_data(args)
    # if input("Check distribution? [Y/n]: ").lower() in ["y", "yes"]:
    data.check_test_dist("Data distribuion of all test data")
    data.check_test_dist_by_client("by_client")

    args.num_classes = data.num_classes
    args.total_num_client, args.test_num_clients = data.train_num_clients, data.test_num_clients
    logger.warn("data.test_num_clients will be deprecated")
    # assert args.total_num_client == args.test_num_clients
    dataset = data.dataset

    # set model
    model = create_model(args, data.input_shape)
    client_selection = client_selection_method(args)
    fed_algo = federated_algorithm(dataset, model, args)

    # save results
    files = utils.save_files(args)

    ## train
    # set federated optim algorithm
    ServerExecute = Server(dataset, model, args, client_selection, fed_algo, files)
    ServerExecute.train()
