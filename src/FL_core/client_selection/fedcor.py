from .client_selection import ClientSelection
import numpy as np
import math
import torch
from copy import deepcopy
from tqdm import tqdm
from itertools import product

from utils import logger

from fedcor.GPR import Kernel_GPR, Matrix_GPR

args.gpr:
        file_name = base_file+'/gpr[int{}_gp{}_norm{}_disc{}]'.\
            format(args.GPR_interval,args.group_size,args.poly_norm,args.discount)
            
            
class FedCor(ClientSelection):
    def __init__(self, args, total, device):
        super().__init__(total, device)
        # Build GP
        if args.kernel=='Poly':
            self.gpr = Kernel_GPR(self.total, loss_type= args.train_method,reusable_history_length=args.group_size,gamma=args.GPR_gamma,device=gpr_device,
                                dimension = args.dimension,kernel=GPR.Poly_Kernel,order = 1,Normalize = args.poly_norm)
        elif args.kernel=='SE':
            self.gpr = Kernel_GPR(self.total, loss_type= args.train_method,reusable_history_length=args.group_size,gamma=args.GPR_gamma,device=gpr_device,
                                dimension = args.dimension,kernel=GPR.SE_Kernel)
        else:
            self.gpr = Matrix_GPR(self.total,loss_type= args.train_method,reusable_history_length=args.group_size,gamma=args.GPR_gamma,device=gpr_device)
        self.gpr.to(device)
        
        self.epsilon_greedy = args.epsilon_greedy,
        self.dynamic_C = args.dynamic_C,
        self.dynamic_TH = args.dynamic_TH
        
    def init(self):
        # copy weights
        global_weights = global_model.state_dict()
        local_weights = []# store local weights of all users for averaging
        local_states = []# store local states of all users, these parameters should not be uploaded

        
        for i in range(self.total):
            local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
            local_weights.append(copy.deepcopy(global_weights))

        local_states = np.array(local_states)
        local_weights = np.array(local_weights)
    
    def select(self, n, client_idxs, metric, round=0, results=None):
        m = max(int(args.frac * self.total), 1)
        
        if self.warmup:
            idxs_users = self.gpr.Select_Clients(
                m, self.epsilon_greedy, weights, 
                self.dynamic_C, self.dynamic_TH)
        else:
            # Random selection
            idxs_users = np.random.choice(range(self.total), m, replace=False)

    def test_gpr(self):
        # test prediction accuracy of GP model
        if self.warmup:
            test_idx = np.random.choice(range(self.total), m, replace=False)
            test_data = np.concatenate([np.expand_dims(list(range(self.total)),1),
                                        np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                        np.ones([self.total,1])],1)
            pred_idx = np.delete(list(range(self.total)),test_idx)
            
            try:
                predict_loss,mu_p,sigma_p = self.gpr.Predict_Loss(test_data,test_idx,pred_idx)
                print("GPR Predict relative Loss:{:.4f}".format(predict_loss))
                predict_losses.append(predict_loss)
            except:
                logger.warn("Singular posterior covariance encountered, skip the GPR test in this round!")
                
    def train_gpr(self):
        if epoch>=args.gpr_begin:
            if epoch<=args.warmup:# warm-up
                gpr.Update_Training_Data([np.arange(args.num_users),],[np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),],epoch=epoch)
                if not args.update_mean:
                    print("Training GPR")
                    gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=150,schedule_lr=False,update_mean=args.update_mean,verbose=args.verbose)
                elif epoch == args.warmup:
                    print("Training GPR")
                    gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=1000,schedule_lr=False,update_mean=args.update_mean,verbose=args.verbose)

            elif epoch>args.warmup and epoch%args.GPR_interval==0:# normal and optimization round
                gpr.Reset_Discount()
                print("Training with Random Selection For GPR Training:")
                random_idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                                    copy.deepcopy(global_model),random_idxs_users,train_dataset,user_groups)
                gpr.Update_Training_Data([np.arange(args.num_users),],[np.array(gpr_loss)-np.array(gt_global_losses[-1]),],epoch=epoch)
                print("Training GPR")
                gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=args.GPR_Epoch,schedule_lr=False,update_mean=args.update_mean,verbose=args.verbose)

            else:# normal and not optimization round
                gpr.Update_Discount(idxs_users,args.discount)
            
        
        if args.mvnt and (epoch==args.warmup or (epoch%args.mvnt_interval==0 and epoch>args.warmup)):
            mvn_file = file_name+'/MVN/{}'.format(seed)
            if not os.path.exists(mvn_file):
                os.makedirs(mvn_file)
            mvn_samples=MVN_Test(args,copy.deepcopy(global_model),
                                        train_dataset,user_groups,
                                        file_name+'/MVN/{}/{}.csv'.format(seed,epoch))
            sigma_gt.append(np.cov(mvn_samples,rowvar=False,bias = True))
            sigma.append(gpr.Covariance().clone().detach().numpy())