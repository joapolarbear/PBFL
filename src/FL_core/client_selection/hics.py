import numpy as np
import copy

from scipy.cluster.hierarchy import linkage, fcluster
import scipy.stats
from itertools import product
from sklearn.cluster import AgglomerativeClustering 
from numpy.random import choice
from copy import deepcopy

from .client_selection import ClientSelection

# refet to: https://github.com/CityChan/HiCS-FL/blob/master/server/server_hics.py

''' Methods copied from the HiCS github '''

def get_similarity(grad_1, grad_2, distance_type="L1"):

    if distance_type == "L1":

        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm

    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return norm

    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)
        
def get_gradients_fc(sampling, global_m, local_models):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""

    local_model_params = []
    for model in local_models:
        local_model_params +=  [
           [tens.detach().cpu().numpy() for tens in list(model.parameters())[-2:]]
        ]
            
    global_model_params = [
        tens.detach().cpu().numpy() for tens in list(global_m.parameters())[-2:]
    ]
    
    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                    local_params, global_model_params
                )
            ]
        ]
    return local_model_grads


def get_clusters_with_alg2(
    linkage_matrix: np.ndarray, 
    n_sampled: int, 
    weights: np.ndarray
):
    """Algorithm 2"""
    epsilon = int(10 ** 10)

    # associate each client to a cluster
    link_matrix_p = deepcopy(linkage_matrix)
    augmented_weights = deepcopy(weights)

    for i in range(len(link_matrix_p)):
        idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])

        new_weight = np.array(
            [augmented_weights[idx_1] + augmented_weights[idx_2]]
        )
        augmented_weights = np.concatenate((augmented_weights, new_weight))
        link_matrix_p[i, 2] = int(new_weight * epsilon)

    clusters = fcluster(
        link_matrix_p, int(epsilon / n_sampled), criterion="distance"
    )

    n_clients, n_clusters = len(clusters), len(set(clusters))

    # Associate each cluster to its number of clients in the cluster
    pop_clusters = np.zeros((n_clusters, 2)).astype(int)
    for i in range(n_clusters):
        pop_clusters[i, 0] = i + 1
        for client in np.where(clusters == i + 1)[0]:
            pop_clusters[i, 1] += int(weights[client] * epsilon * n_sampled)

    pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]

    distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)

    # n_sampled biggest clusters that will remain unchanged
    kept_clusters = pop_clusters[n_clusters - n_sampled :, 0]

    for idx, cluster in enumerate(kept_clusters):
        for client in np.where(clusters == cluster)[0]:
            distri_clusters[idx, client] = int(
                weights[client] * n_sampled * epsilon
            )

    k = 0
    for j in pop_clusters[: n_clusters - n_sampled, 0]:

        clients_in_j = np.where(clusters == j)[0]
        np.random.shuffle(clients_in_j)

        for client in clients_in_j:

            weight_client = int(weights[client] * epsilon * n_sampled)

            while weight_client > 0:

                sum_proba_in_k = np.sum(distri_clusters[k])

                u_i = min(epsilon - sum_proba_in_k, weight_client)

                distri_clusters[k, client] = u_i
                weight_client += -u_i

                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters


def sample_clients(distri_clusters):

    n_clients = len(distri_clusters[0])
    n_sampled = len(distri_clusters)

    sampled_clients = np.zeros(len(distri_clusters), dtype=int)

    for k in range(n_sampled):
        print("cluster ", k)
        print(np.nonzero(distri_clusters[k]))
        sampled_clients[k] = int(choice(n_clients, 1, p=distri_clusters[k]))

    return sampled_clients


class HiCSSelector(ClientSelection):
    def __init__(self, args, total, device):
        super().__init__(total, device)
        
        self.args = args
        
        # Refer to the paper and https://github.com/CityChan/HiCS-FL/blob/master/configs/FMNIST/ablation/FMNIST_hics_1_lambda10_gamma4_M5.json
        # Refert to 
        if args.dataset == "cifar":
            self._temp = 0.015
        elif args.dataset == "fmnist":
            self._temp = 0.0025
        else:
            raise ValueError
        self.hics_alphas = [0.001, 0.002, 0.005, 0.01, 0.5]
        self._lambda = 10
        self._gamma = 4
        self.M = 5
        
        alphas = set()
        for alpha in self.hics_alphas:
            alphas.add(alpha)
        if len(alphas) > 1:
            self.multialpha = True
        else:
            self.multialpha = False

        self.num_round = self.args.num_round
        self.round: int = None
        self.warmup: int = None
        
        self.global_accu = 0
        self.global_loss = 1e6
        
    def before_train(self, n_samples, global_m):
        # Similar to ClusteredSampling2 and PowerOfChoice
        client_ids = sorted(n_samples.keys())
        self.n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = self.n_samples / np.sum(self.n_samples)
        
        self.gradients = get_gradients_fc(
            "clustered_2", 
            global_m, 
            [global_m] * self.total
        )
        self.magnitudes = self._magnitude_gradient(self.gradients)
        
    def before_step(self, global_m, local_models=None):
        self.previous_global_model = copy.deepcopy(global_m)
        
    def select(self, n, client_idxs, metric, round=0, results=None):
        self.round = round
        self.selected_client_num = n
        
        self.warmup = int(self.total / n)
        
        random_pool = list(range(self.total))
        if self.round < self.warmup:
            sampled_clients = random_pool[round*n:(round + 1)*n] 
        else:
            sampled_clients = self._cluster_sampling(
                self.gradients,
                self.magnitudes,
                "cosine",
                round
            )
        return sampled_clients

    def after_step(self, client_idxs, local_models, global_m, loss, acc):
        self.global_loss = loss
        self.global_accu = acc
        
        if self.multialpha:
            if self.round < self.warmup:
                gradients_i = get_gradients_fc(
                    "clustered_2", 
                    self.previous_global_model, local_models
                )
                for idx, gradient in zip(client_idxs, gradients_i):
                    self.gradients[idx] = gradient
        else:
            gradients_i = get_gradients_fc(
                "clustered_2", 
                self.previous_global_model, 
                local_models
            )
            for idx, gradient in zip(client_idxs, gradients_i):
                self.gradients[idx] = gradient
                
        self.magnitudes = self._magnitude_gradient(self.gradients)
    
    def _cluster_sampling(self, gradients, magnitudes, sim_type, round):
        Clusters = []
        n_sampled = max(self.selected_client_num, 1)
        
        magnitudes = self._magnitude_gradient(gradients)
        estimated_H = self._estimated_entropy_from_grad(magnitudes)
        sim_matrix = self._get_matrix_similarity_from_grads_entropy(gradients, estimated_H, distance_type=sim_type)
        linkage_matrix = linkage(sim_matrix, "ward") 

        if np.array(estimated_H).var() < 0.1: 
            hc = AgglomerativeClustering(
                n_clusters=self.M, metric="euclidean", linkage = 'ward'
            ) 
         
            hc.fit_predict(sim_matrix)
            labels = hc.labels_
            for i in range(self.M):
                cluster_i = np.where(labels == i)[0]
                Clusters.append(cluster_i)    
            avg_entropy = self._estimated_entropy(
                estimated_H,Clusters
            )
            return self._sample_clients_entropy(
                avg_entropy, Clusters, self.n_samples, round
            )

        else:
            distri_clusters = get_clusters_with_alg2(
                linkage_matrix, n_sampled, self.weights
            )
            return sample_clients(distri_clusters)
    
    def _sample_clients_entropy(self, entropy, Clusters, n_samples, round):
        n_sampled = max(self.selected_client_num, 1)
        n_clients = len(n_samples)
        n_clustered = len(Clusters)
        entropy = np.exp(self._gamma * (self.num_round - round) * entropy / self.num_round)
    
        p_cluster = entropy/np.sum(entropy)
        sampled_clients = []
        clusters_selected = [0]*n_sampled
        
        for k in range(n_sampled):
            select_group = int(choice(n_clustered, 1, p=p_cluster))
            while clusters_selected[select_group] >= len(Clusters[select_group]):
                select_group = int(choice(n_clustered, 1, p=p_cluster)) 
            clusters_selected[select_group] += 1
        
        for k in range(len(clusters_selected)):
            if clusters_selected[k] == 0:
                continue
            select_clients = choice(
                Clusters[k], clusters_selected[k], replace = False
            )
            for i in range(clusters_selected[k]):
                sampled_clients.append(select_clients[i])
        return sampled_clients

    def _estimated_entropy_from_grad(self, magnitudes):
        estimated_H = []
        T = self._temp
        for idx in range(self.total):
            magnitudes[idx] = np.exp(magnitudes[idx]/T) / np.sum(np.exp(magnitudes[idx]/T))
            pk = np.array(magnitudes[idx])
            estimated_h = scipy.stats.entropy(pk)
            estimated_H.append(estimated_h)
        return estimated_H
    
    def _estimated_entropy(self,estimated_H, Clusters):
        Entropys = []
        for k in range(len(Clusters)):
            print("cluster ", k)
            print(Clusters[k])
            group_entropy = 0
            cluster = Clusters[k]
            for idx in cluster:
                group_entropy += estimated_H[idx]
            if len(cluster) > 0:
                group_entropy = group_entropy/len(cluster)
            Entropys.append(group_entropy)
        Entropys = np.array(Entropys)
        return Entropys
    
    def _get_matrix_similarity_from_grads_entropy(self, local_model_grads, estimated_H,distance_type):
        n_clients = len(local_model_grads)
        metric_matrix = np.zeros((n_clients, n_clients))
        for i, j in product(range(n_clients), range(n_clients)):
            metric = get_similarity(local_model_grads[i], local_model_grads[j], distance_type) 
            metric_matrix[i, j] =  metric + self._lambda*abs(estimated_H[i] - estimated_H[j])
        return metric_matrix
    
    def _magnitude_gradient(self, gradients):
        magnitudes = []
        for idx in range(len(gradients)):
            gradient = gradients[idx][0]
            m, n = gradient.shape
            magnitude = np.zeros(m)
            for c in range(m):
                magnitude[c] = np.sum(gradient[c])/n
            magnitudes.append(magnitude)
        return magnitudes