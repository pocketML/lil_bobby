import torch
import torch.nn as nn

import random, math

from embedding import embeddings

class SelfAttentionWide(nn.Module):
    def __init__(self, cfg, heads=8, mask=False):
        super().__init__()
        
        self.embedding = embeddings.get_embedding(cfg)
        self.heads = heads
        self.mask = mask
        
