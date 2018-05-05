import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .core import Policy
from ..net import SmallCNN, SmallRNN, VAE
from ..util.common import batch_to_sequence, sequence_to_batch
from ..util.init import orthogonal_init


class VAEPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VAEPolicy, self).__init__()
        self.pd = None

        z_size = 128

        self.visual = VAE(z_size, add_noise=False)

        self.mean_head = nn.Linear(z_size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(z_size, 1)

        self.mean_head.apply(orthogonal_init([nn.Linear], 'linear'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'linear'))

    def forward(self, x):
        with torch.no_grad():
            feature, _, _ = self.visual.encode(x)

        mean = self.mean_head(feature)
        std = self.log_std_head.expand_as(mean).exp()

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        value = self.value_head(feature)

        return action, value

    @property
    def recurrent(self):
        return False

    @property
    def name(self):
        return 'VAEPolicy'


class VAELSTMPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(VAELSTMPolicy, self).__init__()
        self.pd = None

        z_size = 128

        self.visual = VAE(z_size, add_noise=False)

        self.rnn = SmallRNN(z_size, z_size)

        self.mean_head = nn.Linear(z_size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(z_size, 1)

        for param in self.visual.parameters():
            param.requires_grad = False

        self.value_head.apply(orthogonal_init([nn.Linear], 'linear'))
        self.mean_head.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        with torch.no_grad:
            feature, _, _ = self.visual.encode(x)

        feature = batch_to_sequence(feature, self.num_steps)
        memory = sequence_to_batch(self.rnn(feature))

        mean = self.mean_head(memory)
        std = self.log_std_head.expand_as(mean).exp()

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        value = self.value_head(memory)

        return action, value

    @property
    def num_steps(self):
        return 1

    @property
    def recurrent(self):
        return True

    @property
    def name(self):
        return 'VAELSTMPolicy'


class CNNPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(CNNPolicy, self).__init__()
        self.pd = None

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.mean_head = nn.Linear(size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(size, 1)

        self.cnn.apply(orthogonal_init([nn.Linear], 'relu'))
        self.mean_head.apply(orthogonal_init([nn.Linear], 'linear'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'linear'))

    def forward(self, x):
        feature = self.cnn(x)

        mean = self.mean_head(feature)
        std = self.log_std_head.expand_as(mean).exp()

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        value = self.value_head(feature)

        return action, value

    @property
    def recurrent(self):
        return False

    @property
    def name(self):
        return 'CNNPolicy'


class CNNLSTMPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(CNNLSTMPolicy, self).__init__()
        self.pd = None

        self.cnn = SmallCNN()

        size = self.cnn.fc.out_features

        self.rnn = SmallRNN(size, size)

        self.mean_head = nn.Linear(size, output_shape[0])
        self.log_std_head = nn.Parameter(torch.ones(output_shape[0]))
        self.value_head = nn.Linear(size, 1)

        self.value_head.apply(orthogonal_init([nn.Linear], 'linear'))
        self.mean_head.apply(orthogonal_init([nn.Linear], 'tanh'))

    def forward(self, x):
        feature = self.cnn(x)

        feature = batch_to_sequence(feature, self.num_steps)
        memory = sequence_to_batch(self.rnn(feature))

        mean = self.mean_head(memory)
        std = self.log_std_head.expand_as(mean).exp()

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        value = self.value_head(memory)

        return action, value

    @property
    def num_steps(self):
        return 4

    @property
    def recurrent(self):
        return True

    @property
    def name(self):
        return 'CNNLSTMPolicy'


class MLPPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(MLPPolicy, self).__init__()
        self.pd = None

        self.pi_fc1 = nn.Linear(input_shape[0], 64)
        self.pi_fc2 = nn.Linear(64, 64)

        self.vf_fc1 = nn.Linear(input_shape[0], 64)
        self.vf_fc2 = nn.Linear(64, 64)

        self.mean_head = nn.Linear(64, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(64, 1)

        self.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.mean_head.apply(orthogonal_init([nn.Linear], 'linear'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'linear'))

    def forward(self, x):
        pi_h1 = F.tanh(self.pi_fc1(x))
        pi_h2 = F.tanh(self.pi_fc2(pi_h1))

        mean = self.mean_head(pi_h2)
        std = self.log_std_head.expand_as(mean).exp()

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        vf_h1 = F.tanh(self.vf_fc1(x))
        vf_h2 = F.tanh(self.vf_fc2(vf_h1))
        value = self.value_head(vf_h2)

        return action, value

    @property
    def recurrent(self):
        return False

    @property
    def name(self):
        return 'MLPPolicy'


class MLPLSTMPolicy(Policy):
    def __init__(self, input_shape, output_shape):
        super(MLPLSTMPolicy, self).__init__()
        self.pd = None

        self.pi_fc = nn.Linear(input_shape[0], 64)
        self.pi_rnn = SmallRNN(64, 64)

        self.vf_fc = nn.Linear(input_shape[0], 64)
        self.vf_rnn = SmallRNN(64, 64)

        self.mean_head = nn.Linear(64, output_shape[0])
        self.log_std_head = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_head = nn.Linear(64, 1)

        self.pi_fc.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.vf_fc.apply(orthogonal_init([nn.Linear], 'tanh'))
        self.value_head.apply(orthogonal_init([nn.Linear], 'linear'))

    def forward(self, x):
        pi_h1 = F.tanh(self.pi_fc(x))
        pi_h1 = batch_to_sequence(pi_h1, self.num_steps)
        pi_h2 = sequence_to_batch(self.pi_rnn(pi_h1))
        pi_h2 = F.tanh(pi_h2)

        mean = self.mean_head(pi_h2)
        std = self.log_std_head.expand_as(mean).exp()

        vf_h1 = F.tanh(self.vf_fc(x))
        vf_h1 = batch_to_sequence(vf_h1, self.num_steps)
        vf_h2 = sequence_to_batch(self.vf_rnn(vf_h1))
        vf_h2 = F.tanh(vf_h2)

        value = self.value_head(vf_h2)

        self.pd = Normal(mean, std)
        action = self.pd.sample() if self.training else mean

        return action, value

    @property
    def num_steps(self):
        return 4

    @property
    def recurrent(self):
        return True

    @property
    def name(self):
        return 'MLPLSTMPolicy'
