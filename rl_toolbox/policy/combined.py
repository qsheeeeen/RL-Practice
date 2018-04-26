import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .core import Policy
from ..net import SmallRNN, MixtureDensityNetwork, VAE
from ..util.distributions import MixtureNormal
from ..util.init import orthogonal_init


class VisualMemoryPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VisualMemoryPolicy, self).__init__()
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
        feature, _, _ = self.visual.encode(x)
        memory = self.memory(feature)

        pi, mean, std = self.mdn(memory)

        self.pd = MixtureNormal(pi, mean, std)
        action = self.pd.sample()

        value = self.value_head(memory)

        return action, value

    @property
    def recurrent(self):
        return True

    @property
    def name(self):
        return 'VisualMemoryPolicy'


class VisualPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VisualPolicy, self).__init__()
        self.pd = None

        z_size = 128
        num_mixture = 5

        self.visual = VAE(z_size)
        self.mdn = MixtureDensityNetwork(z_size, output_shape[0], num_mixture)

        self.mean_head = nn.Linear(z_size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(z_size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))

        for param in self.visual.parameters():
            param.requires_grad = False

        self.visual.apply(orthogonal_init([nn.Linear, nn.Conv2d], 'relu'))
        self.mdn.apply(orthogonal_init([nn.Linear], 'linear'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        feature, _, _ = self.visual.encode(x)

        mean = F.tanh(self.mean_head(feature))
        std = self.log_std_head.expand_as(mean).exp()

        self.pd = Normal(mean, std)
        action = self.pd.sample()

        value = self.value_head(feature)

        return action, value

    @property
    def recurrent(self):
        return False

    @property
    def name(self):
        return 'VisualPolicy'
