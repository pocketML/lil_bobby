import torch
import torch.nn as nn

import math
from tqdm import tqdm
import numpy as np

from compression.distillation.student_models import base
from embedding import embeddings

class WarmupOptimizer:
    """Optim wrapper that implements rate."""

    def __init__(self, base_optimizer, warmup_steps=100, final_lr=1e-4, start_lr=1e-6):
        self.base_optimizer = base_optimizer
        self.warmup_steps = warmup_steps
        self.rates = np.linspace(start_lr, final_lr, num=warmup_steps)
        self.final_lr = final_lr
        self._step = 0
        self._rate = start_lr

    def step(self):
        """Update parameters and rate"""
        self._rate = self.rates[self._step] if self._step < self.warmup_steps else self.final_lr
        self._step += 1
        for p in self.base_optimizer.param_groups:
            p["lr"] = self._rate
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x
    

class EmbFFN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)

        self.pos_encoder = PositionalEncoding(cfg['embedding-dim'], cfg['dropout'])
        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['embedding-dim'], cfg['cls-hidden-dim']),
            nn.Dropout(cfg['dropout']),
            nn.ReLU(),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights()

    def init_weights(self):
        init_range = 0.01
        if self.cfg['embedding-type'] == 'hash':
            self.embedding.init_weight_range(init_range)
        self.classifier[1].bias.data.zero_()
        self.classifier[4].bias.data.zero_()
        self.classifier[1].weight.data.uniform_(-init_range, init_range)
        self.classifier[4].weight.data.uniform_(-init_range, init_range)


    def mean_with_lens(self, x, lens, dim=0):
        if self.cfg['use-gpu']:
            lens = lens.cuda()
        idx = torch.arange(x.shape[1])
        x = x.cumsum(dim)[lens - 1, idx, :]
        x = x / lens.view(-1, 1)
        return x

    def forward(self, x, lens):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        x = self.pos_encoder(x)
        x = self.mean_with_lens(x, lens)
        x = self.classifier(x)
        return x

    def get_optimizer(self):
        warmup_start_lr = self.cfg['lr'] / 100
        base_optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg['lr']        
        )
        optimizer = WarmupOptimizer(
            base_optimizer, 
            warmup_steps=100,
            final_lr=self.cfg['lr'],
            start_lr=warmup_start_lr
        )
        return optimizer