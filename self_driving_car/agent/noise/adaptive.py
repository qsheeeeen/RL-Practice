import torch
import torch.nn.init as init


class AdaptiveNoise(object):
    def __init__(self, params_list):
        self._params_list = params_list

    def apply(self):
        for params in self._params_list:
            x = torch.zeros_like(params.weight)
            init.uniform(x)
            params.weight += x
        raise NotImplementedError
