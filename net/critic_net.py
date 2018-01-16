# coding: utf-8

from torch import nn
from torch.nn import functional


# TODO: Initialize weights from a uniform distribution
class HighDimContinuousNetwork(nn.Module):
    def __init__(self):
        super(HighDimContinuousNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, *input_variable):
        assert len(input_variable) == 1, 'Incorrect input'
        x = functional.relu(self.bn1(self.conv1(input_variable[0])))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        x = functional.relu(self.fc1(x))
        x = functional.tanh(self.fc2(x))

        return x


class HighDimDiscreteNetwork(nn.Module):
    def __init__(self):
        super(HighDimDiscreteNetwork, self).__init__()

    def forward(self, *input_variable):
        pass


class LowDimContinuousNetwork(nn.Module):
    def __init__(self):
        super(LowDimContinuousNetwork, self).__init__()

    def forward(self, *input_variable):
        pass


class LowDimDiscreteNetwork(nn.Module):
    def __init__(self):
        super(LowDimDiscreteNetwork, self).__init__()

    def forward(self, *input_variable):
        pass
