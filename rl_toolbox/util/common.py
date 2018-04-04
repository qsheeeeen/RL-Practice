import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def preprocessing_state(array):
    transform = ToTensor()
    if len(array.shape) == 3:
        tensor = transform(array).float()
    elif len(array.shape) == 1:
        tensor = torch.from_numpy(array).float()
    else:
        raise NotImplementedError

    return tensor.unsqueeze(0)


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
