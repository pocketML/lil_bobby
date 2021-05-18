from analysis import parameters
from common import model_utils

def print_model_size(model):
    total_params, total_bits = parameters.get_model_size(model)
    print(f'total num parameters: {total_params}')
    print(f'size in bits: {total_bits}')
    print(f'size in MBs: {total_bits/8000000:.3f}')
    print('-'*20)

def print_named_params(model, arch):
    layers = model_utils.group_params_by_layer(model, arch)
    for layer, children in layers.items():
        print(f'* {layer}')
        for name, param in children:
            print(f'| --> {name}, {param.size()}, {param.dtype}')


def print_model_disk_size(model):
    size = parameters.get_model_disk_size(model)
    print(f"{size:.3f} MB")

def print_threshold_stats(model, arch):
    layers = model_utils.group_params_by_layer(model, arch)
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    for layer_name, layer in layers.items():
        print(layer_name)
        for threshold in thresholds:
            below, total = count_below_threshold_in_layer(layer, threshold)
            print(f'below {threshold} in {layer_name}: {below}/{total} ({below/total:.4f})')
        print('-'*20)