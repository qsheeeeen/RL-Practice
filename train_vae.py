import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image

from rl_toolbox.policy import VAE, vae_loss

SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPECHS = 10

torch.manual_seed(SEED)

model = VAE()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def load_data():
    # TODO:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.expand(128, 3, 28, 28)
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += vae_loss(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'image/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, EPECHS + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    sample = sample.cuda()
    sample = model.decode(sample).cpu()
    print('Save.')
    save_image(sample.data.view(64, 1, 28, 28),
               'image/sample_' + str(epoch) + '.png')
