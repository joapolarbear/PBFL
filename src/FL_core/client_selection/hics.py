
from .client_selection import ClientSelection

# refet to: https://github.com/CityChan/HiCS-FL/blob/master/server/server_hics.py

''' Methods copied from the HiCS github '''

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



class HiCSSelector(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)
        
        alphas = set()
        for alpha in self.args["alphas"]:
            alphas.add(alpha)

        if len(alphas) > 1:
            self.multialpha = True
        else:
            self.multialpha = False
            
        self.gradients = get_gradients_fc(
            "clustered_2", 
            self.global_model, 
            [self.global_model] * args["n_clients"]
        )
        self.magnitudes = self.magnitude_gradient(self.gradients)
    
    def magnitude_gradient(self, gradients):
        magnitudes = []
        for idx in range(len(gradients)):
            gradient = gradients[idx][0]
            m, n = gradient.shape
            magnitude = np.zeros(m)
            for c in range(m):
                magnitude[c] = np.sum(gradient[c])/n
            magnitudes.append(magnitude)
        return magnitudes