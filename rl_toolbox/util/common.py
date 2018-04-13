import torch
from torch.distributions import Distribution, Normal
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


class MixtureNormal(Distribution):  # TODO: Sample. Test.
    def __init__(self, pi, mean, std):
        self.pi = pi
        self.mean = mean
        self.std = std
        self.m = Normal(self.mean, self.std)

    def sample(self):
        use_cuda = self.pi.is_cuda

        # shape = None
        # z = torch.normal(torch.zero(shape), torch.ones(shape))
        # k = torch.argmax(torch.log(x) + z)
        #
        # indices = (np.arange(n_samples), k)
        # rn = torch.randn(n_samples)
        # sampled = rn * self.std[indices] + self.meanindices]
        # return torch.FloatTensor(sample).cuda() if use_cuda else torch.FloatTensor(sample)

    def sample_n(self, n):
        raise NotImplementedError

    def log_prob(self, value):
        value = value.expand_as(self.mean)
        log_probs = self.m.log_prob(value)
        probs = torch.exp(log_probs)
        weighted_probs = self.pi * probs
        sum_prob = torch.sum(weighted_probs, -1)
        sum_log_prob = torch.log(sum_prob)
        return sum_log_prob
