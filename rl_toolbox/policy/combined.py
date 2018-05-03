import torch
import torch.nn as nn

from .core import Policy
from ..distributions import MixtureNormal
from ..net import SmallRNN, MixtureDensityNetwork, VAE
from ..util.init import orthogonal_init


class VAELSTMPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VAELSTMPolicy, self).__init__()
        self.pd = None

        z_size = 128
        num_mixture = 5

        self.visual = VAE(z_size)
        self.memory = SmallRNN(z_size, z_size)
        self.mdn = MixtureDensityNetwork(z_size, output_shape[0], num_mixture)
        self.value_head = nn.Linear(z_size, 1)

        for param in self.visual.parameters():
            param.requires_grad = False

        self.visual.apply(orthogonal_init([nn.Linear, nn.Conv2d], 'relu'))
        self.memory.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.mdn.apply(orthogonal_init([nn.Linear], 'linear'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        with torch.no_grad:
            feature, _, _ = self.visual.encode(x)
        memory = self.memory(feature)

        pi, mean, std = self.mdn(memory)

        self.pd = MixtureNormal(pi, mean, std)
        action = self.pd.sample()

        value = self.value_head(memory)

        return action, value

    @property
    def num_steps(self):
        return 1

    @property
    def recurrent(self):
        return True

    @property
    def name(self):
        return 'VAELSTMPolicy'


class VAEPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VAEPolicy, self).__init__()
        self.pd = None

        z_size = 128
        num_mixture = 5

        self.visual = VAE(z_size)
        self.mdn = MixtureDensityNetwork(z_size, output_shape[0], num_mixture)
        self.value_head = nn.Linear(z_size, 1)

        for param in self.visual.parameters():
            param.requires_grad = False

        self.visual.apply(orthogonal_init([nn.Linear, nn.Conv2d], 'relu'))
        self.memory.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.mdn.apply(orthogonal_init([nn.Linear], 'linear'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        with torch.no_grad:
            feature, _, _ = self.visual.encode(x)

        pi, mean, std = self.mdn(feature)

        self.pd = MixtureNormal(pi, mean, std)
        action = self.pd.sample()

        value = self.value_head(feature)

        return action, value

    @property
    def num_steps(self):
        return 1

    @property
    def recurrent(self):
        return False

    @property
    def name(self):
        return 'VAEPolicy'
