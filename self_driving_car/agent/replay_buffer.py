from collections import deque

import torch


class ReplayBuffer(object):
    def __init__(self, maxlen):
        self._buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self._buffer)

    def get_all(self):
        if isinstance(self._buffer[0], list):
            return [torch.cat([item[i] for item in self._buffer]).float() for i in range(len(self._buffer[0]))]
        else:
            return torch.cat([item for item in self._buffer], 1).float()

    def store(self, items):
        self._buffer.append(items)

    def clear(self):
        self._buffer.clear()
