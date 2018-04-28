import argparse

import torch

from rl_toolbox.policy import VisualMemoryPolicy, VisualPolicy

parser = argparse.ArgumentParser(description='Train VAE')

# TODO: file name.
parser.add_argument('--vae-path', type=str, default='./weights/vae_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--mdn-path', type=str, default='./weights/mdn_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--rnn-path', type=str, default='./weights/rnn_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--vm-path', type=str, default='./weights/vm_weights.pth', metavar='N',
                    help='where the weights file is.')
parser.add_argument('--v-path', type=str, default='./weights/', metavar='N',
                    help='where the weights file is.')

args = parser.parse_args()

model = VisualMemoryPolicy((96, 96, 3), (3,))
model.mdn.load_state_dict(torch.load(args.mdn_path))
model.vae.load_state_dict(torch.load(args.vae_path))
model.memory.load_state_dict(torch.load(args.rnn_path))
torch.save(model.state_dict(), args.out_path + model.name + '_weights.pth')

model = VisualPolicy((96, 96, 3), (3,))
model.mdn.load_state_dict(torch.load(args.mdn_path))
model.vae.load_state_dict(torch.load(args.vae_path))
torch.save(model.state_dict(), args.out_path + model.name + '_weights.pth')
