import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def init_weights(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.orthogonal(layer.weight, init.calculate_gain('relu'))
        layer.bias.data.fill_(0.)


class CNNBase(nn.Module):
    def __init__(self, input_shape, output_shape):
        assert len(input_shape) == 3, 'Unsupported input shape.'
        assert len(output_shape) == 1, 'Unsupported output shape.'

        super(CNNBase, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc = nn.Linear(1568, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = F.relu(x)

        return x


class CNNPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNNPolicy, self).__init__()

        self.base_model = CNNBase(input_shape, output_shape)

        self.mean_fc = nn.Linear(512, output_shape[0])
        self.std = nn.Parameter(torch.ones(output_shape[0]))
        self.value_fc = nn.Linear(512, 1)

        self.apply(init_weights)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.base_model(x)

        mean = self.mean_fc(x)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(x)

        return mean, std, value


class LSTMPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LSTMPolicy, self).__init__()

        self.base_model = CNNBase(input_shape, output_shape)

        self.rnn = nn.LSTMCell(512, 512)

        self.mean_fc = nn.Linear(512, output_shape[0])
        self.std = nn.Parameter(torch.ones(output_shape[0]))
        self.value_fc = nn.Linear(512, 1)

        self.h, c = None, None

        self.apply(init_weights)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.base_model(x)

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

        self.pi_fc1 = nn.Linear(input_shape[0], 64)
        self.pi_fc2 = nn.Linear(64, 64)

        self.vf_fc1 = nn.Linear(input_shape[0], 64)
        self.vf_fc2 = nn.Linear(64, 64)

        self.mean_fc = nn.Linear(64, output_shape[0])
        self.std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(64, 1)

        self.apply(init_weights)

        self.float()
        self.cuda()

    def forward(self, x):
        pi_h1 = self.pi_fc1(x)
        pi_h1 = F.tanh(pi_h1)

        pi_h2 = self.pi_fc2(pi_h1)
        pi_h2 = F.tanh(pi_h2)

        vf_h1 = self.vf_fc1(x)
        vf_h1 = F.tanh(vf_h1)

        vf_h2 = self.vf_fc2(vf_h1)
        vf_h2 = F.tanh(vf_h2)

        mean = self.mean_fc(pi_h2)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(vf_h2)

        return mean, std, value
