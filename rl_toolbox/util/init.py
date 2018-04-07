import torch.nn.init as init


def orthogonal_init(type_list, nonlinearity='relu', zero_bias=True):
    def init_fn(layer):
        if any([isinstance(layer, layer_type) for layer_type in type_list]):
            init.orthogonal(layer.weight, init.calculate_gain(nonlinearity))
            if zero_bias:
                layer.bias.data.fill_(0.)

    return init_fn
