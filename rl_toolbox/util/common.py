import torch
from torchvision.transforms import ToTensor


def preprocessing_state(array):
    transform = ToTensor()
    dim = array.ndim
    if dim == 3:
        tensor = transform(array)
    elif dim == 1:
        tensor = torch.from_numpy(array).float()
    else:
        raise NotImplementedError('Cant process array with {} dimension(s).'.format(dim))

    return tensor.unsqueeze(0)


def batch_to_sequence(x):
    return x.unsqueeze(1)


def sequence_to_batch(x):
    return x.squeeze(1)
