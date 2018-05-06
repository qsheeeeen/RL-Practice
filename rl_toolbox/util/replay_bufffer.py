from collections import deque

import torch


class ReplayBuffer(object):
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def full(self):
        return len(self.buffer) == self.maxlen

    def get_all(self):
        return [torch.cat([item[i] for item in self.buffer]) for i in range(len(self.buffer[0]))]

    def store(self, items):
        self.buffer.append(items)

    def clear(self):
        self.buffer.clear()
