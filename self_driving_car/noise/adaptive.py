import torch
import torch.nn.init as init


class AdaptiveNoise(object):
    def __init__(self, layers_list, std=0.1, desired_distance=0.1, adoption_coefficient=1.01):
        self._layers_list = layers_list
        self._std = std
        self._desired_distance = desired_distance
        self._adoption_coefficient = adoption_coefficient

    def apply(self):
        for layer in self._layers_list:
            layer.apply(self._apply_noise)

    def adapt(self, distance):
        if distance > self._desired_distance:
            self._std /= self._adoption_coefficient
        else:
            self._std *= self._adoption_coefficient

    def measure_distance(self):
        raise NotImplementedError

    def _apply_noise(self, m):
        x = torch.zeros_like(m.weight.data)
        m.weight.data += init.normal(x, 0, self._std)

        x = torch.zeros_like(m.bias.data)
        m.bias.data += init.normal(x, 0, self._std)
