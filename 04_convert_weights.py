import argparse

import torch

from rl_toolbox.policy import VAELSTMPolicy, VAEPolicy

parser = argparse.ArgumentParser(description='Train VAE')

parser.add_argument('--vae-path', type=str, default='./weights/vae_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--mdn-path', type=str, default='./weights/mdn_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--rnn-path', type=str, default='./weights/rnn_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--out-path', type=str, default='./weights/', metavar='N',
                    help='where the weights file is.')

args = parser.parse_args()

print('Make VAELSTMPolicy weights.')
model = VAELSTMPolicy((96, 96, 3), (3,))
model.visual.load_state_dict(torch.load(args.vae_path))
torch.save(model.state_dict(), args.out_path + model.name + '_weights.pth')

print('Make VAEPolicy weights.')
model = VAEPolicy((96, 96, 3), (3,))
model.visual.load_state_dict(torch.load(args.vae_path))
torch.save(model.state_dict(), args.out_path + model.name + '_weights.pth')

print('Done.')
