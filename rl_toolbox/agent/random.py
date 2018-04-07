import numpy as np

from .core import Agent


class RandomAgent(Agent):
    def __init__(self, input_shape, output_shape, policy):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def act(self, state, reward=0., done=False):
        return np.random.random(self.output_shape)
