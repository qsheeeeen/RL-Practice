import torch.nn as nn
import torch.nn.functional as F

from ..util.init import orthogonal_init


def mdn_loss():  # TODO:
    pass


class MixtureDensityNetwork(nn.Module):  # TODO(2nd): Make this work. how to combine with RNN. How to train.
    def __init__(self, input_shape, output_shape):
        super(MixtureDensityNetwork, self).__init__()

        self.fc1 = nn.Linear(input_shape[0], 512)
        self.fc2 = nn.Linear(512, output_shape[0])

        self.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        h1 = F.tanh(self.fc1(x))
        return F.tanh(self.fc2(h1))
