from compression.distillation.student_models import base
import torch
import torch.nn as nn

import math

from embedding import embeddings

class SelfAttention(nn.Module):
    def __init__(self, emb, num_heads=8, mask=False):
        super().__init__()
        
        assert emb % num_heads == 0, f'Embbeding dimensions ({emb}) should be divisible by heads ({num_heads})'

        self.emb = emb
        self.num_heads = num_heads
        self.mask = mask
        
        self.to_keys = nn.Linear(emb, emb, bias=False)
        self.to_queries = nn.Linear(emb, emb, bias=False)
        self.to_values = nn.Linear(emb, emb, bias=False)

        self.unify_heads = nn.Linear(emb, emb)

    def mask_(self, matrices, mask_val=0.0, mask_diagonal=True):
        _, h, w = matrices.size()
        indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
        matrices[:, indices[0], indices[1]] = mask_val

    def forward(self, x):
        bsz, sqln, emb = x.size()
        num_heads = self.num_heads
        assert emb == self.emb, f'Input embedding dum ({emb}) should match layer embedding dim ({self.emb})'

        # head chunks size
        head_size = emb // num_heads

        ks = self.to_keys(x).view(bsz, sqln, num_heads, head_size)
        qs = self.to_queries(x).view(bsz, sqln, num_heads, head_size)
        vs = self.to_values(x).view(bsz, sqln, num_heads, head_size)

        # fold heads into the batch dimensions
        ks = ks.transpose(1,2).contiguous().view(bsz * num_heads, sqln, head_size)
        qs = qs.transpose(1,2).contiguous().view(bsz * num_heads, sqln, head_size)
        vs = vs.transpose(1,2).contiguous().view(bsz * num_heads, sqln, head_size)

        # instead of dividing the dot product by sqrt(e) we scale the keys and values
        ks = ks / (emb ** (1/4))
        qs = qs / (emb ** (1/4))

        # dot product of queries and keys, and scale
        # bmm = batch matrix-matric multiplication
        dot = torch.bmm(qs, ks.transpose(1,2))

        #print(dot.shape)
        #print(bsz * num_heads, sqln, head_size)
        #assert dot.size() == (bsz * num_heads, sqln, head_size)

        # mask out the uppder half of the dot matrix, excluding the diagonal
        if self.mask:
            self.mask_(dot, mask_val=float('-inf'), mask_diagonal=False)

        # now row-wise self attention probabilities
        dot = nn.functional.softmax(dot, dim=2)

        # apply self attention to the values
        x = torch.bmm(dot, vs).view(bsz, num_heads, sqln, head_size)

        # swap h, t back
        x = x.transpose(1,2).contiguous().view(bsz, sqln, emb)
        x = self.unify_heads(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb, num_heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, num_heads=num_heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = x + attended
        x = self.norm1(x)
        x = self.dropout(x)

        ffed = self.ff(x)
        x = ffed + x
        x = self.norm2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term_sin = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_cos = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term_sin)
        pe[:,1::2] = torch.cos(position * div_term_cos)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x

class Transformer(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg, False)
        # TODO: needs to do some work to add CLS tokens to the different embeddings again
        # TODO: also add support for sentence pairs
        self.cls_token = torch.FloatTensor(self.cfg['embedding-dim'])

        self.embedding_dim_sqrt = math.sqrt(cfg['embedding-dim'])
        self.pos_enc = PositionalEncoding(cfg['embedding-dim'], max_len=cfg['max-tokens'])
        transformers = [TransformerBlock(cfg['embedding-dim'], cfg['attn-heads'], False) for _ in range(cfg['num-layers'])]
        self.encoder = nn.Sequential(*transformers)

        self.decoder = base.get_rnn(cfg)
   
        self.classifier = nn.Sequential(
            nn.Linear(cfg['embedding-dim'], cfg['cls-hidden-dim']),
            nn.Dropout(cfg['dropout']),
            nn.ReLU(),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights(embedding_init_range=0.1, classifier_init_range=0.1)

    def add_cls_(self, x):
        x[:,0,:] = self.cls_token
        return x

    def forward(self, x, _):
        x = self.embedding(x)
        x = self.add_cls_(x) * self.embedding_dim_sqrt
        x = x.permute(1,0,2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.permute(1,0,2)
        x = x[:,0,:]

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