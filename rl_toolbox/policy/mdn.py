import torch.nn as nn
import torch.nn.functional as F

from ..util.common import orthogonal_init


class MixtureDensity(nn.Module):  # TODO(2nd): Make this work. how to combine with RNN. How to train.
    def __init__(self, input_shape, output_shape):
        super(MixtureDensity, self).__init__()

        self.fc = nn.Linear(input_shape[0], 512)

        self.apply(orthogonal_init)

    def forward(self, x):
        return F.relu(self.fc(x))
