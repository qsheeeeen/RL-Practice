# coding: utf-8

import random
from collections import deque

import torch


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        """

        Args:
            buffer_size (int):
        """
        self.buffer = deque(maxlen=buffer_size)
        self.item_len = None

    def __len__(self):
        return len(self.buffer)

    def pop(self, number):
        if len(self.buffer) >= number:
            samples = [self.buffer.pop() for _ in range(number)]

            result = [torch.Tensor([sample[i] for sample in samples]).float() for i in range(len(samples))]

            result = tuple(result)
            return result

        else:
            return None

    def random_sample(self, number):
        if len(self.buffer) >= number:
            samples = random.sample(self.buffer, number)

            result = []
            for i in range(len(samples)):
                result.append(np.array([sample[i] for sample in samples], dtype=np.float32))

            result = tuple(result)
            return result

        else:
            return None

    def store(self, *items):
        self.buffer.append(items)

        self.item_len = len(items)

    def clear(self):
        self.buffer.clear()
