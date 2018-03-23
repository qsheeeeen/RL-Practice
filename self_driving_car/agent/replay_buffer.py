import random
from collections import deque

import torch


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def get_all(self):
        return [torch.cat([item[i] for item in self.buffer]).float() for i in range(len(self.buffer[0]))]

    def random_sample(self, number):
        if len(self.buffer) >= number:
            samples = random.sample(self.buffer, number)
            return [torch.cat([sample[i] for sample in samples]).float() for i in range(len(samples[0]))]
        else:
            return None

    def store(self, items):
        self.buffer.append(items)

    def clear(self):
        self.buffer.clear()
