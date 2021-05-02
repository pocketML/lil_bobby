import torch
import torch.nn as nn

import math

from compression.distillation.student_models import base
from embedding import embeddings

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

        self.embedding = embeddings.get_embedding(cfg, False)

        #self.pos_encoder = PositionalEncoding(cfg['embedding-dim'], cfg['dropout'])
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(3 * cfg['embedding-dim'], cfg['cls-hidden-dim']),
            nn.Dropout(cfg['dropout']),
            nn.ReLU(),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights(0.01)

    def mean_with_lens(self, x, lens, dim=0):
        if self.cfg['use-gpu']:
            lens = lens.cuda()
        idx = torch.arange(x.shape[1])
        x = x.cumsum(dim)[lens - 1, idx, :]
        x = x / lens.view(-1, 1)
        return x

    def cmp(self, x, lens, batch_first=True):
        if self.cfg['use-gpu']:
            lens = lens.cuda()

        x1 = torch.nn.functional.relu(x)
        x2 = self.sigmoid(x)
        if batch_first:
            idx = torch.arange(x.shape[0])
            x1 = x1.cumsum(1)[idx, lens - 1, :]
            x2 = x2.cumsum(1)[idx, lens - 1, :]
            x3 = x.cumsum(1)[idx, lens - 1, :]
        else:
            idx = torch.arange(x.shape[1])
            x1 = x1.cumsum(0)[lens - 1, idx, :]
            x2 = x2.cumsum(0)[lens - 1, idx, :]
            x3 = x.cumsum(0)[lens - 1, idx, :]

        # mean
        x3 = x3 / lens.view(-1, 1)
        return torch.cat([x1, x2, x3], dim=1)

    def forward(self, x, lens):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        #x = self.pos_encoder(x)
        #x = self.sigmoid(x)

        x = self.cmp(x, lens, False) 
        #x = self.mean_with_lens(x, lens)
        x = self.classifier(x)
        return x

    def get_optimizer(self):
        warmup_start_lr = self.cfg['lr'] / 100
        base_optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg['lr']        
        )
        optimizer = base.WarmupOptimizer(
            base_optimizer, 
            warmup_steps=100,
            final_lr=self.cfg['lr'],
            start_lr=warmup_start_lr
        )
        return optimizer