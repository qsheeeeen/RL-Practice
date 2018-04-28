import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, num_mixture):
        super(MixtureDensityNetwork, self).__init__()
        self.num_mixture = num_mixture
        self.output_shape = output_shape

        self.pi_head = nn.Linear(input_shape[0], output_shape[0] * num_mixture)
        self.log_std_head = nn.Linear(input_shape[0], output_shape[0] * num_mixture)
        self.mean_head = nn.Linear(input_shape[0], output_shape[0] * num_mixture)

    def forward(self, x):
        pi_h = self.pi_head(x)
        pi_h = pi_h.view(pi_h.size(0), self.output_shape[0], self.num_mixture)  # TODO: try ConvTranspose2d
        pi = F.softmax(pi_h, -1)

        mean_h = self.mean_head(x)
        mean = mean_h.view(mean_h.size(0), self.output_shape[0], self.num_mixture)

        std_h = self.log_std_head(x).exp()
        std = std_h.view(std_h.size(0), self.output_shape[0], self.num_mixture)

        return pi, mean, std
