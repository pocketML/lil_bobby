import torch
import matplotlib.pyplot as plt
from functools import reduce
from common import model_utils

def dtype_bits(param):
    dt = param.dtype
    if dt in [torch.float32, torch.float, torch.int32, torch.int]:
        return 32
    if dt in [torch.float64, torch.double, torch.complex64, torch.cfloat, torch.int64, torch.long]:
        return 64
    if dt in [torch.complex128, torch.cdouble]:
        return 128
    if dt in [torch.float16, torch.half, torch.bfloat16, torch.int16, torch.short]:
        return 16
    if dt in [torch.uint8, torch.int8]:
        return 8
    raise TypeError(f'Could not recognize dtype: {dt}') 

# returns tuple with number of params and number of bits used
def get_model_size(model):
    total_params = 0
    total_bits = 0
    for _, param in model.named_parameters():
        size = list(param.size())
        num_weights = reduce(lambda acc, x: acc * x, size, 1)
        total_params += num_weights
        # components might have different dtype, so we have to check for each
        total_bits += num_weights * dtype_bits(param)
    return total_params, total_bits

# returns all weights in a layer as a single contiguous 1-dimensional numpy array
def concat_weights_in_layer(layer):
    all_weights = torch.empty(1)
    for name, param in layer:
        if 'weight' in name:
            all_weights = torch.cat((all_weights, torch.flatten(param.contiguous(), 0)), 0)
    return all_weights.detach().numpy()

def count_below_threshold_in_layer(layer, threshold):
    import numpy as np
    weights = concat_weights_in_layer(layer)
    below = np.where(abs(weights) < threshold, 1, 0)
    return below.sum(), len(weights)

def weight_histogram_for_layer(layer, num_bins=1000):
    weights = concat_weights_in_layer(layer)
    plt.hist(weights, bins=num_bins)
    plt.show()

# TODO: only works for RoBERTa models at the moment
def weight_histogram_for_all_transformers(model, arch, num_bins=2000):
    layers = model_utils.group_params_by_layer(model, arch)
    transformers = [layer for layer in layers.keys() if 'layer_' in layer]
    n = len(transformers)
    ncols = 4
    nrows = int(n / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18,15))
    for i, ax in enumerate(axs.flat):
        name = transformers[i]
        layer = layers[name]
        weights = concat_weights_in_layer(layer)
        ax.hist(weights, bins=num_bins)
        ax.set(xlabel='Weight value',ylabel='Frequency', xlim=(-.3,.3), ylim=(0,74000),title=name)
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

def print_threshold_stats(model, arch):
    layers = model_utils.group_params_by_layer(model, arch)
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    for layer_name, layer in layers.items():
        print(layer_name)
        for threshold in thresholds:
            below, total = count_below_threshold_in_layer(layer, threshold)
            print(f'below {threshold} in {layer_name}: {below}/{total} ({below/total:.4f})')
        print('-'*20)

def print_model_size(model):
    total_params, total_bits = get_model_size(model)
    print(f'total num parameters: {total_params}')
    print(f'size in bits: {total_bits}')
    print(f'size in MBs: {total_bits/8000000:.1f}')
    print('-'*20)

def print_named_params(model, arch):
    layers = model_utils.group_params_by_layer(model, arch)
    for layer, children in layers.items():
        print(f'* {layer}')
        for name, param in children:
            print(f'| --> {name}, {param.size()}')
