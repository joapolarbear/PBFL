# config for client selection

class SelectMethod:
    random = "Random"
    gpfl = "GPFL"
    fedcor = "FedCor"
    cluster1 = "Cluster1"
    cluster2 = "Cluster2"
    pow_d = "Pow-d"
    divfl = "DivFL"
    cosin = "Cosin"
    hisc = "HiSC"
    single = "Single"
    afl = "AFL"
    
    
PRE_SELECTION_METHOD = [
    SelectMethod.random, 
    SelectMethod.cluster1,
    SelectMethod.cluster2,
    SelectMethod.gpfl,
    SelectMethod.fedcor,
    SelectMethod.single,
    SelectMethod.cosin,
    SelectMethod.hisc
]


NEED_BEFORE_TRAIN_METHOD = [
    SelectMethod.cluster1,
    SelectMethod.cluster2,
    SelectMethod.pow_d,
    SelectMethod.gpfl,
    SelectMethod.cosin,
    SelectMethod.fedcor,
    SelectMethod.hisc,
]


NEED_BEFORE_STEP_METHOD = [
    SelectMethod.cluster2,
    SelectMethod.divfl,
    SelectMethod.gpfl,
    SelectMethod.cosin,
]

NEED_AFTER_STEP_METHOD = [
    SelectMethod.gpfl, SelectMethod.cosin, SelectMethod.hisc
]


CANDIDATE_SELECTION_METHOD = [SelectMethod.pow_d]


NEED_LOCAL_MODELS_METHOD = [
    'GradNorm', SelectMethod.divfl
]


LOSS_THRESHOLD = ['LossCurr']


CLIENT_UPDATE_METHOD = ['DoCL']


# POST_SELECTION: SelectMethod.pow_d,'AFL','MaxEntropy','MaxEntropySampling','MaxEntropySampling_1_p','MinEntropy',
#                 'GradNorm','GradSim','GradCosSim,SelectMethod.divfl,'LossCurr','MisClfCurr'
