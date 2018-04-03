from collections import deque

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

transform = Compose([
    ToTensor()
])


def processing_image(array):
    if len(array.shape) == 3:
        tensor = transform(array).float()
    elif len(array.shape) == 1:
        tensor = torch.from_numpy(array).float()
    else:
        raise NotImplementedError

    return tensor.unsqueeze(0)


class ReplayBuffer(object):
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def full(self):
        return len(self.buffer) == self.maxlen

    def get_all(self):
        if isinstance(self.buffer[0], list):
            return [torch.cat(item[i] for item in self.buffer) for i in range(len(self.buffer[0]))]
        else:
            return torch.cat([item for item in self.buffer], 1)

    def store(self, items):
        self.buffer.append(items)

    def clear(self):
        self.buffer.clear()


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
