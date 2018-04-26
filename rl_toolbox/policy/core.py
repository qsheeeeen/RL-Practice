import torch.nn as nn


class Policy(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def recurrent(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError
