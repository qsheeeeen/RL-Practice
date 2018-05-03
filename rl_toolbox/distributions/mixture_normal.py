import torch
from torch.distributions import Normal, Categorical


class MixtureNormal(object):
    def __init__(self, pi, loc, scale):
        super(MixtureNormal, self).__init__()
        self.pi = pi
        self.loc = loc
        self.scale = scale

        self.normal_pd = Normal(self.loc, self.scale)
        self.pi_pd = Categorical(self.pi)

    def sample(self):
        with torch.no_grad():
            raw_sample = self.normal_pd.sample()
            index = self.pi_pd.sample().unsqueeze(-1)
            return torch.gather(raw_sample, -1, index).squeeze(-1)

    def log_prob(self, value):
        value = value.unsqueeze(-1).expand_as(self.loc)
        probs = self.normal_pd.log_prob(value).exp()
        weighted_probs = self.pi * probs
        sum_prob = torch.sum(weighted_probs, -1)
        return sum_prob.log()
