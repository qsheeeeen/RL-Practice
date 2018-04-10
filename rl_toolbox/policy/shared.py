import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SmallCNN, RNNBase
from ..util.init import orthogonal_init


class CNNPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNNPolicy, self).__init__()
        self.recurrent = False

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.mean_fc = nn.Linear(size, output_shape[0])
        self.std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))
        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.cnn(x)

        mean = self.mean_fc(feature)
        value = self.value_fc(feature)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        return mean, std, value


class LSTMPolicy(nn.Module):  # TODO(1st): How to train.
    def __init__(self, input_shape, output_shape):
        super(LSTMPolicy, self).__init__()
        self.recurrent = True

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.rnn = RNNBase(size, size)

        self.mean_fc = nn.Linear(size, output_shape[0])
        self.std = nn.Parameter(torch.ones(output_shape[0]))
        self.value_fc = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))
        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))
        self.rnn.apply(orthogonal_init([nn.Linear], 'tanh'))

        self.hidden = None

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.cnn(x)

        memory = self.rnn(feature)

        mean = self.mean_fc(memory)
        value = self.value_fc(memory)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        return mean, std, value


class MLPPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MLPPolicy, self).__init__()
        self.recurrent = False

        self.pi_fc1 = nn.Linear(input_shape[0], 64)
        self.pi_fc2 = nn.Linear(64, 64)

        self.vf_fc1 = nn.Linear(input_shape[0], 64)
        self.vf_fc2 = nn.Linear(64, 64)

        self.mean_fc = nn.Linear(64, output_shape[0])
        self.log_std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(64, 1)

        self.apply(orthogonal_init([nn.Linear], 'tanh'))

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

        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(vf_h2)

        return mean, std, value
