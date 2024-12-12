import argparse
from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'log': (None, 'None for no logging'),
        'lr': (0.001, 'learning rate'),
        'batch-size': (10000, 'batch size'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0.005, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
        'device': (0, 'which device'),
    },
    'model_config': {
        'model': ('VorRec', 'model name'),
        'embedding_dim': (64, 'user item embedding dimension'),
        'scale': (0.1, 'scale for init'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'num-layers': (3,  'number of hidden layers in encoder'),
        'margin1': (0.1, 'margin value in the Voronoi loss'),
        'margin2': (0.1, 'margin value in the recommend loss'),
        'optim': ('rsgd', 'optimizer choice'),
        'lambda1': (1e-1, 'lambda for Voronoi loss'),
        'lambda2': (1e-1, 'lambda for contrastive learning loss'),
        'tau': (0.1, 'temperature for contrastive learning'),
    },
    'data_config': {
        'dataset': ('ciao', 'which dataset to use'),
        'num_neg': (1, 'number of negative samples'),
        'test_ratio': (0.2, 'proportion of test edges for link prediction'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
