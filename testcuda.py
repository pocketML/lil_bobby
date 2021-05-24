import torch
import numpy as np 
import random 

seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


a = torch.ones(1000,1000).to('cuda:0')
dropout = torch.nn.Dropout(0.5).cuda()
b = dropout(a)

print(torch.sum(torch.abs(b)))