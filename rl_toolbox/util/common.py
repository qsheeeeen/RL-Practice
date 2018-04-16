import torch
from torch.distributions import Distribution, Normal, Categorical
from torch.utils.data import Dataset
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


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class MixtureNormal(Distribution):  # TODO: Test.
    def __init__(self, pi, mean, std):
        assert all([tensor.dim() == 3 for tensor in (pi, mean, std)])

        self.pi = pi if pi.dim() > 2 else pi.unsqueeze(0)
        self.mean = mean
        self.std = std
        self.pd = Normal(self.mean, self.std)
        self.pi_pd = [Categorical(prob) for prob in self.pi]

    def sample(self):
        raw_sample = self.pd.sample()
        index = torch.stack([pd.sample() for pd in self.pi_pd]).unsqueeze(-1)
        return torch.gather(raw_sample, -1, index).squeeze(-1)

    def sample_n(self, n):
        raise NotImplementedError

    def log_prob(self, value):
        value = value.unsqueeze(-1).expand_as(self.mean)
        log_probs = self.pd.log_prob(value)
        probs = torch.exp(log_probs)
        weighted_probs = self.pi * probs
        sum_prob = torch.sum(weighted_probs, -1)
        return torch.log(sum_prob)
