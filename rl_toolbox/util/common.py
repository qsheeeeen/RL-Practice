import torch
from torch.distributions import Distribution, Normal, Categorical
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


class MixtureNormal(Distribution):
    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        raise NotImplementedError

    def __init__(self, pi, loc, scale):
        assert all([tensor.dim() == 3 for tensor in (pi, loc, scale)]), 'Current only support 3 dims.'
        super(MixtureNormal, self).__init__()

        self.pi = pi
        self.loc = loc
        self.scale = scale
        self.pd = Normal(self.mean, self.std)
        self.pi_pd = [Categorical(prob) for prob in self.pi]

    def sample(self, sample_shape=torch.Size()):
        raw_sample = self.pd.sample()
        index = torch.stack([pd.sample() for pd in self.pi_pd]).unsqueeze(-1)
        return torch.gather(raw_sample, -1, index).squeeze(-1)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        value = value.unsqueeze(-1).expand_as(self.mean)
        log_probs = self.pd.log_prob(value)
        probs = torch.exp(log_probs)
        weighted_probs = self.pi * probs
        sum_prob = torch.sum(weighted_probs, -1)
        return torch.log(sum_prob)

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    @property
    def _nature_params(self):
        pass
