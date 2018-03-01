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
        samples = [torch.stack([item[i] for item in self.buffer]).float() for i in range(len(self.buffer[0]))]
        self.buffer.clear()

        return samples

    def pop(self, number):
        if len(self.buffer) >= number:
            samples = [self.buffer.pop() for _ in range(number)]

            return [torch.stack([sample[i] for sample in samples]).float() for i in range(len(samples[0]))]

        else:
            return None

    def random_sample(self, number):
        if len(self.buffer) >= number:
            samples = random.sample(self.buffer, number)

            return [torch.stack([sample[i] for sample in samples]).float() for i in range(len(samples[0]))]

        else:
            return None

    def store(self, items):
        self.buffer.append(items)

    def clear(self):
        self.buffer.clear()
