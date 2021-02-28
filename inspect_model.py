from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa # load the Commonsense QA task
from functools import reduce
import torch

roberta = RobertaModel.from_pretrained('./models/checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
roberta.eval() # disable dropout

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

def group_by_layer(model):
    layers = {}
    for name, param in roberta.named_parameters():
        if 'classification_heads' in name:
            key = name.split('classification_heads.')[1].split('.')[0]
        elif 'layers.' in name:
            key = 'layer_' + (name.split('layers.')[1].split('.')[0])
        else:
            key = 'encoding_layer'
        values = layers.get(key, [])
        values.append((name,param))
        layers[key] = values
    return layers

def print_parameter_size(model):
    total_weights = 0
    total_params = 0
    total_bits = 0
    total_mbs = 0
    for name, param in model.named_parameters():
        size = list(param.size())
        num_weights = reduce(lambda acc, x: acc * x, size, 1)
        num_bits = num_weights * dtype_bits(param)

        total_weights += num_weights
        total_bits += num_bits
        total_mbs += num_bits / 8_000_000
        total_params += 1

    print(f'Number of weights: {total_weights}')
    print(f'Number of named params: {total_params}')
    print('---------------')
    print('Size of weights:')
    print(f'{total_bits} bits')
    print(f'{total_mbs:.2f} MB')

def map_inplace(param, op):
    sizes = list(param.size())
    if len(sizes) > 1:
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                value = param[i,j]
                param[i,j] = op(value)
    else:
        for i in range(sizes[0]):
            value = param[i]
            param[i] = op(value)

def fold(param, op, acc):
    sizes = list(param.size())
    if len(sizes) > 1:
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                value = param[i, j]
                op(value, acc)
    else:
        for i in range(sizes[0]):
            value = param[i]
            op(value, acc)
    return acc

#layers = group_by_layer(roberta)
#ratio_below_treshold(roberta, 0.0001)
#print() .data?

layers = group_by_layer(roberta)
'''
threshold = 0.01
def count_below(v, acc):
    if abs(v) < 0.01:
        acc[1] += 1
    acc[0] += 1
print(f'Setting threshold to: {threshold}')
for key in layers.keys():
    if 'layer_' in key:
        print(key)
        total = 0
        below = 0
        result = [0,0]
        for name,param in layers[key]:
            if 'weight' in name:
                result = fold(param, count_below, result)
        print(f'below threshold: {result[1]}/{result[0]} ({result[1]/result[0]:.4f})')
'''


print('-'*20)
print_parameter_size(roberta)