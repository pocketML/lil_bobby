import torch
import matplotlib.pyplot as plt
from functools import reduce
from common import model_utils
import os
import numpy as np

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

def get_model_disk_size(model, sacred_experiment=None):
    temp_name = "tmp.pt" if sacred_experiment is None else f"tmp_{sacred_experiment.info['name']}.pt"
    torch.save(model.state_dict(), temp_name)
    size = os.path.getsize(temp_name)/1e6
    os.remove(temp_name)
    return size

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

def get_theoretical_size(model):
    nonzero_params = 0
    nonzero_bits = 0
    for _, param in model.named_parameters():
        non_zero = torch.count_nonzero(param).item()
        nonzero_params += non_zero
        # components might have different dtype, so we have to check for each
        nonzero_bits += non_zero * dtype_bits(param)
    return nonzero_params, nonzero_bits
