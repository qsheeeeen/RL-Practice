import numpy as  np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.init import orthogonal_init


def gaussian_distribution(y, mean, std):
    # normalization factor for Gaussians
    oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0 * np.pi)

    # make |mean|=K copies of y, subtract mean, divide by std
    result = (y.expand_as(mean) - mean) / std
    result = -0.5 * pow(result, 2)
    return (torch.exp(result) / std) * oneDivSqrtTwoPI


def mdn_loss(pi, std, mean, y):
    result = gaussian_distribution(y, mean, std) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


class MixtureDensityNetwork(nn.Module):  # TODO(2nd): Make this work. how to combine with RNN. How to train.
    def __init__(self, input_shape, output_shape):
        super(MixtureDensityNetwork, self).__init__()

        self.fc1 = nn.Linear(input_shape[0], 512)

        self.pi_fc = nn.Linear(512, output_shape[0])
        self.log_std_fc = nn.Linear(512, output_shape[0])
        self.mean_fc = nn.Linear(512, output_shape[0])

        self.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        h1 = F.tanh(self.fc1(x))

        pi = F.softmax(self.pi_fc(h1), -1)
        mean = self.mean_fc(h1)
        std = torch.exp(self.log_std_fc(h1))

        return pi, mean, std
