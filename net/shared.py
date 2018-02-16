# coding: utf-8

from torch import nn
from torch.nn import functional
from torch.distributions import Normal  # TODO: LogNormal


class SharedNetwork(nn.Module):
    def __init__(self, use_cuda=True):
        super(SharedNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=5, padding=2),
        self.bn_1 = nn.BatchNorm2d(16),
        self.relu_1 = nn.ReLU(),
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=5, padding=2),
        self.bn_2 = nn.BatchNorm2d(32),
        self.relu_2 = nn.ReLU()

        self.fc = nn.Linear(7 * 7 * 32, 128)

        self.mean_fc = nn.Linear(128, 2)
        self.std_fc = nn.Linear(128, 2)
        self.value_fc = nn.Linear(128, 1)

        self.m = None

        if use_cuda:
            self.cuda()

    def forward(self, x):
        out = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        out = functional.relu(functional.max_pool2d(self.conv2(out), 2))
        out = out.view(-1, 320)
        out = functional.relu(self.fc1(out))
        out = functional.dropout(out, training=self.training)
        out = self.fc2(out)

        mean = self.mean_fc(out)
        mean = functional.tanh(mean)

        std = self.std_fc(out)

        value = self.value_fc(out)
        value = functional.tanh(value)

        self.m = Normal(mean, std)
        action = self.m.sample()

        return action, value

    def log_prob(self, state, value):
        self.forward(state)
        return self.m.log_prob(value)
