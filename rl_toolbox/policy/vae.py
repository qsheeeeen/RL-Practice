import torch.nn as nn
from torch.autograd import Variable

from .shared import SmallCNN
from ..util.common import orthogonal_init


def vae_loss():
    pass


class VAE(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(VAE, self).__init__()
        self.recurrent = False

        self.encoder = SmallCNN()

        self.apply(orthogonal_init([nn.Linear, nn.Conv2d], 'relu'))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.float()
        self.cuda()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
