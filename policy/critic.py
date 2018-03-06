# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, num_outputs):
        super(CriticNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)

        self.fc_1 = nn.Linear(294912, 256)
        self.fc_2 = nn.Linear(256, 128)

        self.value_fc = nn.Linear(128, 1)

        self.float()
        self.cuda()

    def forward(self, s, a):
        x = self.conv_1(s)
        x = self.bn_1(x)
        x = F.tanh(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.tanh(x)

        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.tanh(x)
        x = self.fc_2(x)
        x = F.tanh(x)

        value = self.value_fc(x)

        return value
