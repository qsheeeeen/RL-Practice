# coding: utf-8

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.autograd import Variable
from visualdl import LogWriter

from .core import Agent
from .replay_buffer import ReplayBuffer


# TODO
class PPOAgent(Agent):
    def __init__(self):
        raise NotImplementedError

    def act(self, observation, reward, done):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
