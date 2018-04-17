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
        self.pd = None

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.mean_head = nn.Linear(size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))
        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.cnn(x)

        mean = self.mean_head(feature)
        log_std = self.log_std_head.expand_as(mean)
        std = torch.exp(log_std)

        value = self.value_head(feature)

        self.pd = Normal(mean, std)
        action = self.pd.sample()

        return action, value

    def log_prob(self, x):
        return self.pd.log_prob(x)


class CNNLSTMPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(CNNLSTMPolicy, self).__init__()
        self.recurrent = True
        self.pd = None

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.rnn = SmallRNN(size, size)

        self.mean_head = nn.Linear(size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.ones(output_shape[0]))
        self.value_head = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear], 'linear'))
        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))
        self.rnn.apply(orthogonal_init([nn.Linear], 'tanh'))

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.cnn(x)

        memory = self.rnn(feature)

        mean = self.mean_head(memory)
        value = self.value_head(memory)

        log_std = self.log_std_head.expand_as(mean)
        std = torch.exp(log_std)

        self.pd = Normal(mean, std)
        action = self.pd.sample()

        return action, value

    def log_prob(self, x):
        return self.pd.log_prob(x)


class MLPPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(MLPPolicy, self).__init__()
        self.recurrent = False
        self.pd = None

        self.pi_fc1 = nn.Linear(input_shape[0], 64)
        self.pi_fc2 = nn.Linear(64, 64)

        self.vf_fc1 = nn.Linear(input_shape[0], 64)
        self.vf_fc2 = nn.Linear(64, 64)

        self.mean_head = nn.Linear(64, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(64, 1)

        self.apply(orthogonal_init([nn.Linear], 'tanh'))

        self.float()
        self.cuda()

    def forward(self, x):
        pi_h1 = F.tanh(self.pi_fc1(x))
        pi_h2 = F.tanh(self.pi_fc2(pi_h1))

        # NOTE: try tanh
        mean = self.mean_head(pi_h2)
        log_std = self.log_std_head.expand_as(mean)
        std = torch.exp(log_std)

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        vf_h1 = F.tanh(self.vf_fc1(x))
        vf_h2 = F.tanh(self.vf_fc2(vf_h1))
        value = self.value_head(vf_h2)

        return action, value

    def log_prob(self, x):
        return self.pd.log_prob(x)


class MLPLSTMPolicy(Policy):  # Note: Try single rnn layer
    def __init__(self, input_shape, output_shape):
        super(MLPLSTMPolicy, self).__init__()
        self.recurrent = True
        self.pd = None

        self.pi_fc = nn.Linear(input_shape[0], 64)
        self.pi_rnn = SmallRNN(64, 64)

        self.vf_fc = nn.Linear(input_shape[0], 64)
        self.vf_rnn = SmallRNN(64, 64)

        self.mean_head = nn.Linear(64, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(64, 1)

        self.apply(orthogonal_init([nn.Linear], 'tanh'))

        self.float()
        self.cuda()

    def forward(self, x):
        pi_h1 = F.tanh(self.pi_fc(x))
        pi_h2 = F.tanh(self.pi_rnn(pi_h1))

        mean = self.mean_head(pi_h2)
        log_std = self.log_std_head.expand_as(mean)
        std = torch.exp(log_std)

        vf_h1 = F.tanh(self.vf_fc(x))
        vf_h2 = F.tanh(self.vf_rnn(vf_h1))
        value = self.value_head(vf_h2)

        self.pd = Normal(mean, std)
        action = self.pd.sample()

        return action, value

    def log_prob(self, x):
        return self.pd.log_prob(x)
