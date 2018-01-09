# coding: utf-8

import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape):
        self.buffer = deque(maxlen=buffer_size)

    def get_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done))

    def erase(self):
        self.buffer.clear()
