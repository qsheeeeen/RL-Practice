import torch.nn as nn


class Policy(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError
