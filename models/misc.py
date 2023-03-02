import copy

import torch.nn as nn
from torch.nn.modules.container import ModuleList


def build_mlp(input_dim, hidden_dims, output_dim=None, use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])