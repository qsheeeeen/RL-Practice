import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityNetwork(nn.Module):  # TODO(2nd): Test.
    def __init__(self, input_shape, output_shape, num_mixture):
        super(MixtureDensityNetwork, self).__init__()
        self.num_mixture = num_mixture
        self.output_shape = output_shape

        self.fc1 = nn.Linear(input_shape[0], 512)

        self.pi_fc = nn.Linear(512, output_shape[0])
        self.log_std_fc = nn.Linear(512, output_shape[0])
        self.mean_fc = nn.Linear(512, output_shape[0])

    def forward(self, x):
        h1 = F.tanh(self.fc1(x))

        pi_h = self.pi_fc(h1)
        pi_h = pi_h.view(pi_h.size(0), self.output_shape[0], self.num_mixture)  # TODO: try ConvTranspose2d
        pi = F.softmax(pi_h, -1)

        mean_h = self.mean_fc(h1)
        mean = mean_h.view(mean_h.size(0), self.output_shape[0], self.num_mixture)

        log_std_h = self.log_std_fc(h1)
        log_std = log_std_h.view(log_std_h.size(0), self.output_shape[0], self.num_mixture)
        std = torch.exp(log_std)

        return pi, mean, std
