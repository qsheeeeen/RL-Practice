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

    def get_all(self):
        results = [[item[i] for item in self.buffer] for i in range(len(self.buffer[0]))]
        return [torch.stack(thing) for thing in results]

    def pop(self, number):
        """
        Notes:
            Return is in reversed order.

        Args:
            number:

        Returns:

        """
        if len(self.buffer) >= number:
            samples = [self.buffer.pop() for _ in range(number)]

            result = [torch.Tensor([sample[i] for sample in samples]).float() for i in range(len(samples))]

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

    def store(self, items):
        self.buffer.append(items)

    def clear(self):
        self.buffer.clear()
