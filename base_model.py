import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):

    def __init__(self, args, dataset):
        super(BaseModel, self).__init__()

        # load dataset info
        self.feature_mapping = dataset.feature_mapping
        self.feature_names = dataset.feature_names

        # load parameters info

        # define network

    def calculate_loss(self, *args):
        raise NotImplementedError()

    def evaluate(self, *args):
        raise NotImplementedError()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        prefix = '-------- model info --------\n'
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return prefix + super().__str__() + '\nTrainable parameters: {}'.format(params)
