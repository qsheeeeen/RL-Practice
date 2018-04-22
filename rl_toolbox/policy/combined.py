import torch
import torch.nn as nn
from torch.distributions import Normal

from .core import Policy
from ..net import SmallRNN, MixtureDensityNetwork, VAE
from ..util.common import MixtureNormal
from ..util.init import orthogonal_init


class VisualMemoryPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VisualMemoryPolicy, self).__init__()
        self.name = 'VisualMemoryPolicy'
        self.recurrent = True
        self.pd = None

        z_size = 128
        num_mixture = 5

        self.visual = VAE(z_size)
        self.memory = SmallRNN(z_size, z_size)
        self.mdn = MixtureDensityNetwork(z_size, output_shape[0], num_mixture)
        self.value_head = nn.Linear(z_size, 1)

        for param in self.visual.parameters():
            param.requires_grad = False

        self.value_head.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.memory.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.mdn.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        feature, _, _ = self.visual.encode(x)
        memory = self.memory(feature)

        pi, mean, std = self.mdn(memory)
        self.pd = MixtureNormal(pi, mean, std)
        action = self.pd.sample()

        value = self.value_head(memory)

        return action, value


class VisualPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VisualPolicy, self).__init__()
        self.name = 'VisualMemoryPolicy'
        self.recurrent = False
        self.pd = None

        z_size = 128

        self.visual = VAE(z_size)

        self.mean_head = nn.Linear(z_size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(z_size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))

        for param in self.visual.parameters():
            param.requires_grad = False

    def forward(self, x):
        feature, _, _ = self.visual.encode(x)

        mean = self.mean_head(feature)
        log_std = self.log_std_head.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_head(feature)

        self.pd = Normal(mean, std)
        action = self.pd.sample()

        return action, value
