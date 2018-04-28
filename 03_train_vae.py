import argparse

import h5py
import numpy as np
import torch
import torch.utils.data
import torch.utils.data
from torch import optim
from torch.utils.data.dataset import TensorDataset, random_split
from torchvision.utils import save_image

from rl_toolbox.net.vae import VAE, vae_loss
from rl_toolbox.util.common import preprocessing_state

parser = argparse.ArgumentParser(description='Train VAE')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--weights-path', type=str, default='./weights/vae_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--data-path', type=str, default='./data/CarRacing-v0-CNNPolicy.hdf5', metavar='N',
                    help='where the hdf5 file is.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')


def get_data_loader(file_path):
    print('Open file.')
    with h5py.File(file_path, 'r') as file:
        samples = np.array(file['states'])

    print('Processing data.')
    data_t = torch.cat([preprocessing_state(sample) for sample in samples])

    test_length = int(len(data_t) * 0.2)
    train_length = len(data_t) - test_length

    print('Make dataset.')
    data_set = TensorDataset(data_t, torch.empty(len(data_t)))

    train_dataset, test_dataset = random_split(data_set, (train_length, test_length))

    print('Make data loader.')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    return train_loader, test_loader


train_loader, test_loader = get_data_loader(args.data_path)

print('Make model.')
model = VAE().to(device)

# if args.load:
#     model.load_state_dict(torch.load(args.load_path))

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 3, 96, 96)[:n]])
                save_image(comparison.cpu(),
                           'image/vae_reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))


for e in range(1, args.epochs + 1):
    train(e)
    test(e)
    # with torch.no_grad():
    #     sample = torch.randn(64, 128).to(device)
    #     sample = model.decode(sample).cpu()
    #     save_image(sample,
    #                'image/vae_sample_' + str(e) + '.png')

torch.save(model.state_dict(), args.weights_path)
