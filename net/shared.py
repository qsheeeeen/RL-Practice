# coding: utf-8

import torch
from torch import nn
from torch.nn import functional


class SharedNetwork(nn.Module):
    def __init__(self, use_cuda=True):
        super(SharedNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)

        self.fc_1 = nn.Linear(294912, 256)
        self.fc_2 = nn.Linear(256, 128)

        self.mean_fc = nn.Linear(128, 2)
        self.std_fc = nn.Linear(128, 2)
        self.std = nn.Parameter(torch.zeros(2))
        self.value_fc = nn.Linear(128, 1)

        if use_cuda:
            self.cuda()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = functional.tanh(x)

        if (x != x).any():
            raise ValueError('Find nan in x.')

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = functional.tanh(x)

        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = functional.tanh(x)
        x = self.fc_2(x)
        x = functional.tanh(x)

        mean = self.mean_fc(x)

        # std = self.std_fc(x)

        std = self.std.expand_as(mean)
        std = torch.exp(std)

        value = self.value_fc(x)

        return mean, std, value
