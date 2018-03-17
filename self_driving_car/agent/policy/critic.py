# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCritic(nn.Module):
    def __init__(self, num_outputs):
        super(CNNCritic, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(65536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128))

        self.action_fc = nn.Linear(128, num_outputs)
        self.value_fc = nn.Linear(128, 1)

        self.float()
        self.cuda()

    def forward(self, x, a):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        value = self.value_fc(x)

        return value


class MLPCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPCritic, self).__init__()

        self.fc_1 = nn.Linear(num_inputs, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 128)

        self.mean_fc = nn.Linear(128, num_outputs)
        self.std = nn.Parameter(torch.zeros(num_outputs))
        self.value_fc = nn.Linear(128, 1)

        self.float()
        self.cuda()

    def forward(self, x, a):
        x = self.fc_1(x)
        x = F.tanh(x)
        x = self.fc_2(x)
        x = F.tanh(x)
        x = self.fc_3(x)
        x = F.tanh(x)

        value = self.value_fc(x)

        return value
