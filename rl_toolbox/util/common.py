import torch
import torch.nn.init as init
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def orthogonal_init(type_list, nonlinearity='relu', zero_bias=True):
    def init_fn(layer):
        if any([isinstance(layer, layer_type) for layer_type in type_list]):
            init.orthogonal(layer.weight, init.calculate_gain(nonlinearity))
            if zero_bias:
                layer.bias.data.fill_(0.)

    return init_fn


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


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
