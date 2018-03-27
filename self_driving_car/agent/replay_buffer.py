import torch


class ReplayBuffer(object):
    def __init__(self):
        self._buffer = []

    def __len__(self):
        return len(self._buffer)

    def get_all(self):
        return [torch.cat([item[i] for item in self._buffer]).float() for i in range(len(self._buffer[0]))]

    def store(self, items):
        self._buffer.append(items)

    def clear(self):
        del self._buffer[:]
