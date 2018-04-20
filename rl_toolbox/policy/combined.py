import torch.nn as nn

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

        self.visual = VAE(z_size)
        self.memory = SmallRNN(z_size, z_size)
        self.mdn = MixtureDensityNetwork(z_size, output_shape[0], 5)
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
