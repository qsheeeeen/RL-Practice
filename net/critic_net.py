# coding: utf-8

from torch import nn, cuda
from torch.nn import functional


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(423, 256)
        self.fc2 = nn.Linear(256, 1)

        self.float()

        if cuda.is_available():
            print('Using GPU.')
            self.cuda()
        else:
            print('Using CPU.')
            self.cpu()

    def forward(self, *input_variable):
        assert len(input_variable) == 2, 'Incorrect input'
        image_input = functional.relu(self.bn1(self.conv1(input_variable[0])))
        image_input = functional.relu(self.bn2(self.conv2(image_input)))
        image_input = functional.relu(self.bn3(self.conv3(image_input)))

        x = functional.relu(self.fc1(image_input))

        reward_input = input[0]

        return self.fc2(x)
