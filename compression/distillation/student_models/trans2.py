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

class Transformer2(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)
        self.embedding_dim_sqrt = math.sqrt(cfg['embedding-dim'])
        self.cls_token = torch.FloatTensor(self.cfg['embedding-dim'])
        self.cls_token.uniform_(-cfg['emb-init-range'], cfg['emb-init-range'])

        self.pos_encoder = PositionalEncoding(cfg['embedding-dim'], cfg['dropout'])
        encoder_layers = nn.TransformerEncoderLayer(
            cfg['embedding-dim'], 
            cfg['attn-heads'],
            cfg['attn-hidden'],
            cfg['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, cfg['num-layers'])

        self.classifier = nn.Sequential(
            nn.Linear(cfg['embedding-dim'], cfg['cls-hidden-dim']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights()

    def add_cls_(self, x):
        x[:,0,:] = self.cls_token
        return x

    def forward(self, x, x_mask):
        x = self.embedding(x)
        x = self.add_cls_(x) * self.embedding_dim_sqrt
        x = x.permute(1,0,2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[0,:,:]
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