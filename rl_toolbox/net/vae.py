import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.init import orthogonal_init


def vae_loss(recon_x, x, mu, sigma):
    bce = F.mse_loss(recon_x, x, size_average=False)
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return bce + kl_divergence


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()

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


class _Decoder(nn.Module):
    def __init__(self):
        super(_Decoder, self).__init__()
        self.fc = nn.Linear(512, 4096)

        self.convt1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.convt3 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4)

    def forward(self, x):
        h1 = F.relu(self.fc(x))

        h1 = h1.view(h1.size(0), 64, 8, 8)

        h2 = F.relu(self.convt1(h1))
        h3 = F.relu(self.convt2(h2))

        return F.sigmoid(self.convt3(h3))


class VAE(nn.Module):
    def __init__(self, z_size=128, add_noise=True):
        super(VAE, self).__init__()
        self.add_noise = add_noise

        self.encoder = _Encoder()
        self.decoder = nn.Sequential(nn.Linear(128, 512), _Decoder())

        encoder_output_shape = self.encoder.fc.out_features

        self.mu_fc = nn.Linear(encoder_output_shape, z_size)
        self.sigma_fc = nn.Linear(encoder_output_shape, z_size)

        self.apply(orthogonal_init([nn.Linear, nn.Conv2d, nn.ConvTranspose2d], 'relu'))

    def encode(self, x):
        feature = self.encoder(x)
        mu = self.mu_fc(feature)
        sigma = self.sigma_fc(feature)

        z = mu + sigma * torch.randn_like(mu) if (self.add_noise and self.training) else mu

        return z, mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, sigma = self.encode(x)
        return self.decode(z), mu, sigma
