import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SmallCNN, SmallCNNTranspose
from ..util.init import orthogonal_init


def vae_loss(recon_x, x, mu, sigma):
    bce = F.mse_loss(recon_x, x, size_average=False)
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return bce + kl_divergence


class VAE(nn.Module):
    def __init__(self, z_size=128):
        super(VAE, self).__init__()
        self.recurrent = False

        self.encoder = SmallCNN()
        self.decoder = nn.Sequential(nn.Linear(128, 512), SmallCNNTranspose())

        encoder_output_shape = self.encoder.fc.out_features

        self.mu_fc = nn.Linear(encoder_output_shape, z_size)
        self.sigma_fc = nn.Linear(encoder_output_shape, z_size)

        self.apply(orthogonal_init([nn.Linear, nn.Conv2d], 'relu'))

        self.float()
        self.cuda()

    def encode(self, x):
        feature = self.encoder(x)
        mu = self.mu_fc(feature)
        sigma = self.sigma_fc(feature)

        if self.training:
            z = mu + sigma * torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        else:
            z = mu

        return z, mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, sigma = self.encode(x)
        return self.decode(z), mu, sigma
