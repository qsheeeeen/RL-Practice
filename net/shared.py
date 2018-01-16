# coding: utf-8

from torch import nn
from torch.nn import functional


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()

    def forward(self, *input_variable):
        pass
