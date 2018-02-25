# coding: utf-8

import torch
from torch import nn
from torch.nn import functional
from torch.distributions import Normal  # TODO: LogNormal


class SharedNetwork(nn.Module):
    def __init__(self, use_cuda=True):
        super(SharedNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)

        self.fc_1 = nn.Linear(294912, 256)
        self.fc_2 = nn.Linear(256, 128)

        self.mean_fc = nn.Linear(128, 3)
        self.std_fc = nn.Linear(128, 3)
        self.value_fc = nn.Linear(128, 1)

        self.m = None

        if use_cuda:
            self.cuda()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = functional.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = functional.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = functional.relu(x)
        x = self.fc_2(x)
        x = functional.relu(x)

        mean = self.mean_fc(x)
        mean = functional.tanh(mean)

        std = self.std_fc(x)
        std = torch.exp(std)

        value = self.value_fc(x)

        self.m = Normal(mean, std)
        action = self.m.sample()

        return action, value

    def log_prob(self, action):
        return self.m.log_prob(action)
