# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    def __init__(self, num_outputs):
        super(CNNPolicy, self).__init__()

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # self.bn2d_1 = nn.BatchNorm2d(64)
        # self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=7, padding=2)
        # self.bn2d_2 = nn.BatchNorm2d(128)
        # self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=5, padding=1)
        # self.bn2d_3 = nn.BatchNorm2d(256)

        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.bn2d_4 = nn.BatchNorm2d(256)

        self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.bn2d_5 = nn.BatchNorm2d(256)
        # self.pool_5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc_1 = nn.Linear(92416, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_3 = nn.Linear(512, 128)

        self.mean_fc = nn.Linear(128, num_outputs)
        self.std = nn.Parameter(torch.zeros(num_outputs))
        self.value_fc = nn.Linear(128, 1)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.conv_1(x)
        # x = self.bn2d_1(x)
        x = F.tanh(x)
        # x = self.pool_1(x)

        x = self.conv_2(x)
        # x = self.bn2d_2(x)
        x = F.tanh(x)
        # x = self.pool_2(x)

        x = self.conv_3(x)
        # x = self.bn2d_3(x)
        x = F.tanh(x)

        x = self.conv_4(x)
        # x = self.bn2d_4(x)
        x = F.tanh(x)

        x = self.conv_5(x)
        # x = self.bn2d_5(x)
        x = F.tanh(x)
        # x = self.pool_5(x)

        x = x.view(x.size(0), -1)

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


class MLPPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPPolicy, self).__init__()

        self.fc_1 = nn.Linear(num_inputs, 256)
        # self.bn_1 = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(256, 256)
        # self.bn_2 = nn.BatchNorm1d(256)
        self.fc_3 = nn.Linear(256, 128)
        # self.bn_3 = nn.BatchNorm1d(128)

        self.mean_fc = nn.Linear(128, num_outputs)
        self.std = nn.Parameter(torch.zeros(num_outputs))
        self.value_fc = nn.Linear(128, 1)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.fc_1(x)
        # x = self.bn_1(x)
        x = F.tanh(x)

        x = self.fc_2(x)
        # x = self.bn_2(x)
        x = F.tanh(x)

        x = self.fc_3(x)
        # x = self.bn_3(x)
        x = F.tanh(x)

        mean = self.mean_fc(x)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(x)

        return mean, std, value
