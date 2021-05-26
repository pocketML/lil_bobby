import torch

import random
import numpy as np

def set_global_seed(seed, set_cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if set_cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

SEED_DICT = {
    "bennington": 3201976,
    "hadfield": 8291959,
    "feynman": 5111918,
    "simone": 2211933
    #"knuth": 1101938,
    #"shannon": 4301916,
    #"miyazaki": 1051941,
    #"doom": 7131971,
    #"lorca": 6051898,
    #"armstrong": 8051930,
    #"cardi": 10111992,
    #"lovelace": 12101815,
    #"elite": 1337
}