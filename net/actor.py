# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, num_outputs):
        super(ActorNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)

        self.fc_1 = nn.Linear(294912, 256)
        self.fc_2 = nn.Linear(256, 128)

        self.action_fc = nn.Linear(128, num_outputs)

        self.float()
        self.cuda()

    def forward(self, state):
        x = self.conv_1(state)
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

        action = self.action_fc(x)

        return action
