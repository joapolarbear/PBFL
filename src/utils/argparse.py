import argparse

ALL_METHODS = [
    'Random', 'Cluster1', 'Cluster2', 'Pow-d', 'AFL', 'DivFL', 'GradNorm',
    'GPFL', "FedCorr", "Single", "FedCor", "Cosin"
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST', help='dataset',
                        # choices=['Reddit','FederatedEMNIST','FedCIFAR100','CelebA', 'PartitionedCIFAR10', 'FederatedEMNIST_IID', 'FederatedEMNIST_nonIID']
                        )
    parser.add_argument('--data_dir', type=str, default='../dataset/FederatedEMNIST', help='dataset directory')
    parser.add_argument('--model', type=str, default='CNN', help='model',
        # choices=['BLSTM','CNN','ResNet']
    )
    parser.add_argument('--method', type=str, default='Random', help='client selection',
                        choices=ALL_METHODS)
    parser.add_argument('--fed_algo', type=str, default='FedAvg', help='Federated algorithm for aggregation',
                        choices=['FedAvg', 'FedAdam'])
    
    # optimizer
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.1, help='learning rate for client optim')
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for server optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')
    parser.add_argument('--beta', type=float, default=0)
    
    parser.add_argument('--lr_decay',type = float,default=0.1,
                        help = 'Learning rate decay at specified rounds')
    parser.add_argument('--schedule', type=int, nargs='*', default=[162, 244],
                        help='Decrease learning rate at these rounds.')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha1', type=float, default=0.75, help='alpha1 for AFL')
    parser.add_argument('--alpha2', type=float, default=1, help='alpha2 for AFL')
    parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3 for AFL')
    
    # training setting
    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    parser.add_argument('-R', '--num_round', type=int, default=2000, help='total number of rounds')
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    parser.add_argument('-K', '--total_num_clients', type=int, default=None, help='total number of clients')

    parser.add_argument('-u', '--num_updates', type=int, default=None, help='number of updates')
    parser.add_argument('-n', '--num_available', type=int, default=None, help='number of available clients at each round')
    parser.add_argument('-d', '--num_candidates', type=int, default=None, help='buffer size; d of power-of-choice')

    parser.add_argument('--loss_div_sqrt', action='store_true', default=False, help='loss_div_sqrt')
    parser.add_argument('--loss_sum', action='store_true', default=False, help='sum of losses')
    parser.add_argument('--num_gn', type=int, default=0, help='number of group normalization')

    parser.add_argument('--distance_type', type=str, default='L1', help='distance type for clustered sampling 2')
    parser.add_argument('--subset_ratio', type=float, default=0.1, help='subset size for DivFL')

    parser.add_argument('--dirichlet_alpha', type=float, default=None, help='ratio of data partition from dirichlet distribution')
    
    parser.add_argument('--min_num_samples', type=int, default=None, help='mininum number of samples')
    # parser.add_argument('--schedule', type=int, nargs='+', default=[0, 5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300],
    #                     help='splitting points (epoch number) for multiple episodes of training')
    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    # Additional model arguments for models in FedCor
    parser.add_argument('--kernel_sizes', type=int, default=[3, 3, 3],nargs="*",
                        help='kernel size in each convolutional layer')
    parser.add_argument('--num_filters', type=int, default=[32, 64, 64],nargs = "*",
                        help="number of filters in each convolutional layer.")
    parser.add_argument('--padding', action='store_true', 
                        help='use padding in each convolutional layer')
    parser.add_argument('--mlp_layers',type= int,default=[64,],nargs="*",
                        help="numbers of dimensions of each hidden layer in MLP, or fc layers in CNN")
    parser.add_argument('--depth',type = int,default = 20, 
                        help = "The depth of ResNet. Only valid when model is resnet")

    # Additional model arguments for FedCor
    parser.add_argument('--discount',type = float, default=0.9, 
                        help = 'annealing coefficient, i.e., beta in paper')
    parser.add_argument('--GPR_interval',type = int, default= 5, 
                        help = 'interval of sampling and training of GP, namely, Delta t')
    parser.add_argument('--GPR_gamma',type = float,default = 0.8,
                        help = 'gamma for training GP')
    parser.add_argument('--GPR_Epoch',type=int,default=100,
                        help = 'number of optimization iterations of GP')
    parser.add_argument('--verbose', type=int, default=0, 
                        help='verbose')
    parser.add_argument('--update_mean', action='store_true', 
                        help="Whether to update the mean of the GPR")
    parser.add_argument('--warmup',type = int, default=25,
                        help='length of warm up phase for GP')
    parser.add_argument('--poly_norm',type=int,default=0,
                        help='whether to normalize the poly kernel, set 1 to normalize')
    parser.add_argument('--group_size',type = int, default=10, 
                        help='length of history round to sample for GP, equal to M in paper')
    parser.add_argument('--kernel',type = str,default = 'Poly',
                        help = 'kind of kernel used in GP (Poly,SE)')
    parser.add_argument('--train_method',type = str,default='MML',
                        help = 'method of training GP (MML,LOO)')
    parser.add_argument('--dimension',type = int,default=15,
                        help = 'dimension of embedding in GP')
    parser.add_argument('--mu',type = float, default=0.0,
                        help = 'mu in FedProx')
    
    # Additional arguments for data distribution
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unbalanced data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--shards_per_client',type = int,default=1,
                        help='number of shards for each client')
    
    # experiment setting
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')
    parser.add_argument('--save_probs', action='store_true', default=False, help='save probs')
    parser.add_argument('--no_save_results', action='store_true', default=False, help='save results')
    parser.add_argument('--test_freq', type=int, default=1, help='test all frequency')
    
    # UCB exploration parameter
    parser.add_argument('--ucb_alpha', type=str, default='round_', help='UCB exploration parameter alpha, with the following types'
                        '1): --ucb_alpha=const_a, alpha = a'
                        '2): --ucb_alpha=linear_a, alpha = a * step'
                        '3): --ucb_alpha=round_, alpha= step / 100 if step < 100 else 200 / (step + 100)')

    # Others
    parser.add_argument('--comment', type=str, default='', help='comment')
    args = parser.parse_args()
    return args