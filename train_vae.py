import argparse

import h5py
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.utils.data.dataset import random_split
from torchvision.utils import save_image

from rl_toolbox.policy import VAE, vae_loss
from rl_toolbox.util.common import preprocessing_state, TensorDataset

parser = argparse.ArgumentParser(description='Train VAE')
parser.add_argument('--data-path', type=str, default='./data/1525.hdf5', metavar='N',
                    help='where the hdf5 file is.')
parser.add_argument('--load', action='store_true', default=True,
                    help='load trained weights.')
parser.add_argument('--load-path', type=str, default='./weights/vae_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

file = h5py.File(args.data_path, 'r')

data = np.array(file['states'])

data_t = torch.cat([preprocessing_state(data) for data in data])

length = len(data_t)

data_set = TensorDataset(data_t, data_t)

train_dataset, test_dataset = random_split(data_set, (length - 1000, 1000))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size, shuffle=True)

model = VAE()

if args.load:
    model.load_state_dict(torch.load(args.load_path))

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
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
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += vae_loss(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 3, 96, 96)[:n]])
            save_image(comparison.data.cpu(),
                       'image/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    torch.save(model.state_dict(), './weights/vae_weights.pth')
    # sample = Variable(torch.randn(64, 20))
    # if args.cuda:
    #     sample = sample.cuda()
    # sample = model.decode(sample).cpu()
    # save_image(sample.data.view(64, 1, 28, 28),
    #            'results/sample_' + str(epoch) + '.png')
