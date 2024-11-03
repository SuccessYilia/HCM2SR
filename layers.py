# -*- coding: utf-8 -*-
# Time  : 2022/4/8
# Author: slmu
# Email : mushanlei.msl@alibaba-inc.com

import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


class DiscreteEmbeddingLayers(nn.Module):
    def __init__(self, feature_mapping, discrete_feature_names):
        super(DiscreteEmbeddingLayers, self).__init__()
        self.feature_mapping = feature_mapping
        self.discrete_feature_names = discrete_feature_names
        self.embedding_dict = dict()
        self.total_dim = 0
        for feature in discrete_feature_names:
            n_features = int(feature_mapping[feature]['ValueNum'])
            embedding_size = int(feature_mapping[feature]['EmbeddingDim'])
            self.embedding_dict[feature] = nn.Embedding(n_features + 1, embedding_size, padding_idx=n_features)
            self.total_dim += embedding_size

        self.embedding_dict = nn.ModuleDict(self.embedding_dict)
        self.apply(xavier_normal_initialization)

    def forward(self, input_tensor):
        input_embeddings = []
        for idx, feature in enumerate(self.discrete_feature_names):
            input_embeddings.append(self.embedding_dict[feature](input_tensor[:, idx]))
        input_embeddings = torch.cat(input_embeddings, dim=1)
        return input_embeddings


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0., activation='leakyrelu', bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(xavier_normal_initialization)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name='relu'):
    activation = None
    if isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
