import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .core import Policy
from ..net.common import SmallCNN, SmallRNN
from ..util.init import orthogonal_init


class CNNPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(CNNPolicy, self).__init__()
        self.recurrent = False
        self.pd_fn = Normal

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.mean_fc = nn.Linear(size, output_shape[0])
        self.log_std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))
        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.cnn(x)

        mean = self.mean_fc(feature)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_fc(feature)

        return mean, std, value


class CNNLSTMPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(CNNLSTMPolicy, self).__init__()
        self.recurrent = True

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.rnn = SmallRNN(size, size)

        self.mean_fc = nn.Linear(size, output_shape[0])
        self.log_std = nn.Parameter(torch.ones(output_shape[0]))
        self.value_fc = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))
        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))
        self.rnn.apply(orthogonal_init([nn.Linear], 'tanh'))

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.cnn(x)

        memory = self.rnn(feature)

        mean = self.mean_fc(memory)
        value = self.value_fc(memory)

        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        return mean, std, value


class MLPPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(MLPPolicy, self).__init__()
        self.recurrent = False
        self.pd = None

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
        pi_h1 = F.tanh(self.pi_fc1(x))
        pi_h2 = F.tanh(self.pi_fc2(pi_h1))

        mean = self.mean_fc(pi_h2)
        # TODO: try tanh
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        vf_h1 = F.tanh(self.vf_fc1(x))
        vf_h2 = F.tanh(self.vf_fc2(vf_h1))
        value = self.value_fc(vf_h2)

        self.pd = Normal(mean, std)
        action = self.pd.sample()

        return action, value

    def log_prob(self, x):
        return self.pd.log_prob(x)


class MLPLSTMPolicy(Policy):  # Note: Try single rnn layer
    def __init__(self, input_shape, output_shape):
        super(MLPLSTMPolicy, self).__init__()
        self.recurrent = True
        self.pd_fn = Normal

        self.pi_fc = nn.Linear(input_shape[0], 64)
        self.pi_rnn = SmallRNN(64, 64)

        self.vf_fc = nn.Linear(input_shape[0], 64)
        self.vf_rnn = SmallRNN(64, 64)

        self.mean_fc = nn.Linear(64, output_shape[0])
        self.log_std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(64, 1)

        self.apply(orthogonal_init([nn.Linear], 'tanh'))

        self.float()
        self.cuda()

    def forward(self, x):
        pi_h1 = F.tanh(self.pi_fc(x))
        pi_h2 = F.tanh(self.pi_rnn(pi_h1))

        mean = self.mean_fc(pi_h2)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        vf_h1 = F.tanh(self.vf_fc(x))
        vf_h2 = F.tanh(self.vf_rnn(vf_h1))
        value = self.value_fc(vf_h2)

        return mean, std, value
