import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.common import orthogonal_init


class CNNBase(nn.Module):
    def __init__(self, input_shape, output_shape):
        assert len(input_shape) == 3, 'Unsupported input shape.'
        assert len(output_shape) == 1, 'Unsupported output shape.'

        super(CNNBase, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(4096, 512)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))

        h3 = h3.view(h3.size(0), -1)

        return F.relu(self.fc(h3))


class CNNPolicy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNNPolicy, self).__init__()

        self.base_model = CNNBase(input_shape, output_shape)

        size = self.base_model.fc.out_features

        self.mean_fc = nn.Linear(size, output_shape[0])
        self.std = nn.Parameter(torch.zeros(output_shape[0]))
        self.value_fc = nn.Linear(size, 1)

        self.apply(orthogonal_init([nn.Linear, nn.Conv2d], 'relu'))

        self.float()
        self.cuda()

    def forward(self, x):
        size = self.base_model(x)

        mean = self.mean_fc(size)
        value = self.value_fc(size)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

        return mean, std, value


class LSTMPolicy(nn.Module):  # TODO(1st): Make this thing work. Why slow. How to train.
    def __init__(self, input_shape, output_shape, batch_size):
        super(LSTMPolicy, self).__init__()

        self.base_model = CNNBase(input_shape, output_shape)

        self.size = self.base_model.fc.out_features

        # self.rnn = nn.LSTMCell(size, size)
        self.rnn = nn.LSTM(self.size, self.size)

        self.mean_fc = nn.Linear(self.size, output_shape[0])
        self.std = nn.Parameter(torch.ones(output_shape[0]))
        self.value_fc = nn.Linear(self.size, 1)

        self.apply(self.base_model.init_weights)

        self.batch_size = batch_size

        self.hidden = None

        self.float()
        self.cuda()

    def forward(self, x):
        feature = self.base_model(x)

        out, self.hidden = (self.rnn(feature) if self.hidden is None else self.rnn(feature, self.hidden))

        mean = self.mean_fc(out)
        value = self.value_fc(out)

        log_std = self.std.expand_as(mean)
        std = torch.exp(log_std)

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

    def save(self, weight_path):
        print('Save weights.')
        torch.save(self.state_dict(), weight_path)

    def load(self, weight_path):
        print('Load weights.')
        self.load_state_dict(torch.load(weight_path))
