import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CNNPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        assert len(input_shape) == 3, 'Unsupported input shape.'
        assert len(output_shape) == 1, 'Unsupported output shape.'

        super(CNNPolicy, self).__init__()

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc = nn.Linear(1568, 512)

        self.mean_fc = nn.Linear(512, output_shape[0])
        self.std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(512, 1)

        # self.apply(init.xavier_normal)  # TODO

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
    def __init__(self, input_shape, output_shape):
        assert len(input_shape) == 1, 'Unsupported input shape.'
        assert len(output_shape) == 1, 'Unsupported output shape.'

        super(MLPPolicy, self).__init__()

        self.fc_1 = nn.Linear(input_shape[0], 64)
        self.fc_2 = nn.Linear(64, 64)

        self.mean_fc = nn.Linear(64, output_shape[0])
        self.std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(64, 1)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.fc_1(x)
        x = F.tanh(x)

        x = self.fc_2(x)
        x = F.tanh(x)

        mean = self.mean_fc(x)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(x)

        return mean, std, value
