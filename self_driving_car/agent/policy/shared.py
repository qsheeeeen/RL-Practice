# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    def __init__(self, num_outputs):
        super(CNNPolicy, self).__init__()

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc = nn.Linear(3200, 256)

        self.mean_fc = nn.Linear(256, num_outputs)
        self.std = nn.Parameter(torch.zeros(num_outputs))
        self.value_fc = nn.Linear(256, 1)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = F.relu(x)

        mean = self.mean_fc(x)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(x)

        return mean, std, value


class MLPPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPPolicy, self).__init__()

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.fc_3 = nn.Linear(256, 128)

        self.mean_fc = nn.Linear(128, num_outputs)
        self.std = nn.Parameter(torch.zeros(num_outputs))
        self.value_fc = nn.Linear(128, 1)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.fc_1(x)
        x = F.tanh(x)

        x = self.fc_2(x)
        x = F.tanh(x)

        x = self.fc_3(x)
        x = F.tanh(x)

        mean = self.mean_fc(x)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(x)

        return mean, std, value
