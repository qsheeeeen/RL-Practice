# coding: utf-8

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.autograd import Variable

from .core import Agent
from .replay_buffer import ReplayBuffer


# TODO
class PPOAgent(Agent):
    def __init__(self):
        pass

    def act(self, observation, reward, done):
        pass

    def save(self):
        pass

    def load(self):
        pass
