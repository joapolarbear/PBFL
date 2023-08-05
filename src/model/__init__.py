import torch
from torchvision.models import resnet18
from .BLSTM import BLSTM
from .CNN import CNN_DropOut, CNN, CNN_CIFAR_dropout, ModelCNNCeleba
from .resnet_gn import resnet18, resnet34, resnet50, resnet101, resnet152

import fedcor

def create_model(args, input_shape):
    if args.dataset in ["cifar", "mnist", "fmnist"]:
        # BUILD MODEL
        _model_type = args.model.lower()
        if _model_type == 'cnn':
            # Naive Convolutional neural netork
            model = fedcor.models.NaiveCNN(args, input_shape, final_pool=False)
        elif _model_type == "resnet18":
            model = resnet18()
        elif _model_type == 'bncnn':
            # Convolutional neural network with batch normalization
            model = fedcor.models.BNCNN(args, input_shape)
        elif _model_type == 'mlp' or _model_type == 'log':
            # Multi-layer preceptron
            len_in = 1
            for x in input_shape:
                len_in *= x
                model = fedcor.models.MLP(dim_in=len_in, dim_hidden=args.mlp_layers if _model_type=='mlp' else [],
                                dim_out=args.num_classes)
        elif _model_type == 'resnet':
            model = fedcor.models.ResNet(args.depth, args.num_classes)
        elif _model_type == 'rnn':
            if args.dataset=='shake':
                model = fedcor.models.RNN(256, args.num_classes)
            else:
                # emb_arr,_,_= get_word_emb_arr('./data/sent140/embs.json')
                model = fedcor.models.RNN(256, args.num_classes,300,True,128)
        else:
            exit('Error: unrecognized model')
    elif args.dataset == 'Reddit' and args.model == 'BLSTM':
        model = BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)
    elif args.dataset == 'FederatedEMNIST_nonIID' and args.model == 'CNN':
        model = CNN_DropOut(True)
    elif args.dataset == 'FederatedEMNIST_nonIID' and args.model == 'CNN':
        model = CNN_DropOut(True)
    elif 'FederatedEMNIST' in args.dataset and args.model == 'CNN':
        model = CNN_DropOut(False)
    elif args.dataset == 'FedCIFAR100' and args.model == 'ResNet':
        model = resnet18(num_classes=args.num_classes, group_norm=args.num_gn)  # ResNet18+GN
    elif args.dataset == 'CelebA' and args.model == 'CNN':
        model = ModelCNNCeleba()
    elif args.dataset == 'PartitionedCIFAR10':
        model = CNN_CIFAR_dropout()

    model = model.to(args.device)
    if args.parallel:
        model = torch.nn.DataParallel(model, output_device=0)
    return model

