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

class Transformer2(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg, False)
        self.embedding_dim_sqrt = math.sqrt(cfg['embedding-dim'])

        self.pos_encoder = PositionalEncoding(cfg['embedding-dim'], cfg['dropout'])
        encoder_layers = nn.TransformerEncoderLayer(
            cfg['embedding-dim'], 
            cfg['attn-heads'],
            cfg['attn-hidden'],
            cfg['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, cfg['transformer-layers'])

        self.decoder = base.get_rnn(cfg)

        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['encoder-hidden-dim'] * 2, cfg['cls-hidden-dim']),
            nn.Dropout(cfg['dropout']),
            nn.ReLU(),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights(embedding_init_range=0.1, classifier_init_range=0.1)

    def apply_mask(self, x):
        mask = torch.LongTensor([0,0,0])
        bitmask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < self.cfg['train-masking']
        if self.cfg['use-gpu']:
            mask = mask.cuda()
        idx = bitmask.nonzero()
        x[idx[:,0], idx[:,1], :] = mask
        return x

    def forward(self, x, lens):
        #if self.training and self.cfg['train-masking'] > 0:
        #    x = self.apply_mask(x)
        #print(x.shape)
        x = self.embedding(x)
        #print(x.shape)
        x = x.permute(1,0,2)
        #print(x.shape)
        x = self.pos_encoder(x)
        #print(x.shape)
        x = self.transformer_encoder(x)
        #print(x.shape)
        x = x.permute(1,0,2)
        #print(x.shape)
        
        h = base.pack_rnn_unpack(self.decoder, self.cfg, x, lens, x.shape[0])
        x = base.choose_hidden_state(h, lens=lens, decision='last') 
        #print(x.shape)
        

        #x = x.mean(dim=0)
        x = self.classifier(x)
        #print(x.shape)
        return x

    def get_optimizer(self):
        lr = self.cfg['lr']
        warmup_start_lr = lr / 100
        base_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = WarmupOptimizer(
            base_optimizer, 
            warmup_steps=100,
            final_lr=lr,
            start_lr=warmup_start_lr
        )
        optimizer = base_optimizer
        return optimizer
