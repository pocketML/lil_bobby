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

total_weights = 0
total_params = 0
total_bits = 0
total_mbs = 0
for name, param in roberta.named_parameters():
    size = list(param.size())
    num_weights = reduce(lambda acc, x: acc * x, size, 1)
    num_bits = num_weights * dtype_bits(param)

    total_weights += num_weights
    total_bits += num_bits
    total_mbs += num_bits / 8_000_000
    total_params += 1
    print(name, list(param.size()))

print('******************')
print(f'Number of weights: {total_weights}')
print(f'Number of named params: {total_params}')
print('---------------')
print('Size of weights:')
print(f'{total_bits} bits')
print(f'{total_mbs:.2f} MB')