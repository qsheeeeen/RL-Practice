# coding: utf-8

import h5py
import numpy as np
from pyglet.window import key

from .core import Agent


class KeyboardAgent(Agent):
    def __init__(self):
        self.file = h5py.File('./data.h5', 'w')

        self.state_data_set = self.file.create_dataset('state', (10000, 96, 96, 3), np.float32, chunks=(1, 96, 96, 3))
        self.action_data_set = self.file.create_dataset('action', (10000, 3), np.float32, chunks=(1, 3))

        self.action_array = np.zeros((3,), np.float32)

        self.count = 0

    def act(self, state, reward=0, done=False):
        self.state_data_set[self.count] = state
        self.action_data_set[self.count] = self.action_array

        self.count += 1

        return self.action_array

    def close(self):
        self.file.close()

    def load(self):
        raise NotImplementedError

    def key_press(self, k, mod):
        if k == key.LEFT:
            self.action_array[0] = -1.0
        if k == key.RIGHT:
            self.action_array[0] = +1.0
        if k == key.UP:
            self.action_array[1] = +1.0
        if k == key.DOWN:
            self.action_array[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(self, k, mod):
        if k == key.LEFT and self.action_array[0] == -1.0:
            self.action_array[0] = 0
        if k == key.RIGHT and self.action_array[0] == +1.0:
            self.action_array[0] = 0
        if k == key.UP:
            self.action_array[1] = 0
        if k == key.DOWN:
            self.action_array[2] = 0
