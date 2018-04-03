import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CNNBase(nn.Module):
    def __init__(self, input_shape, output_shape):
        assert len(input_shape) == 3, 'Unsupported input shape.'
        assert len(output_shape) == 1, 'Unsupported output shape.'

        super(CNNBase, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc = nn.Linear(3200, 256)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))

        h2 = h2.view(h2.size(0), -1)

        return F.relu(self.fc(h2))


# TODO: why not converge...
class CNNPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNNPolicy, self).__init__()

        self.base_model = CNNBase(input_shape, output_shape)

        feature = self.base_model.fc.out_features

        self.mean_fc = nn.Linear(feature, output_shape[0])
        self.std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(feature, 1)

        # self.apply(self.init_weights)

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.base_model(x)

        mean = self.mean_fc(feature)
        value = self.value_fc(feature)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        return mean, std, value

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.orthogonal(layer.weight, init.calculate_gain('relu'))
            layer.bias.data.fill_(0.)


# TODO: why extremely slow...
class LSTMPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LSTMPolicy, self).__init__()

        self.base_model = CNNBase(input_shape, output_shape)

        self.rnn = nn.LSTMCell(128, 128)

        self.hidden = None

        self.mean_fc = nn.Linear(128, output_shape[0])
        self.std = nn.Parameter(torch.ones(output_shape[0]))
        self.value_fc = nn.Linear(128, 1)

        self.apply(self.base_model.init_weights)

        self.float()
        self.cuda()

    def forward(self, x):
        x = self.base_model(x)

        if self.hidden is None:
            self.hidden = self.init_hidden(x)

        self.hidden = self.rnn(x, self.hidden)

        self.hidden[0].detach_()
        self.hidden[1].detach_()

        x = self.hidden[0]

        mean = self.mean_fc(x)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(x)

        return mean, std, value

    @staticmethod
    def init_hidden(x):
        return torch.zeros_like(x), torch.zeros_like(x)


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

        self.apply(self._init_weights)

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

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Linear):
            init.orthogonal(layer.weight, init.calculate_gain('tanh'))
            layer.bias.data.fill_(0.)
