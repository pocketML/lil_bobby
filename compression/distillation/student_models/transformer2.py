import torch
import torch.nn as nn

import math
from tqdm import tqdm

from compression.distillation.student_models import base
from embedding import embeddings

class WarmupOptimizer:
    """Optim wrapper that implements rate."""

    def __init__(self, base_optimizer, d_model, scale_factor, warmup_steps):
        self.base_optimizer = base_optimizer
        self.warmup_steps = warmup_steps
        self.scale_factor = scale_factor
        self.d_model = d_model
        self._step = 1
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        self._rate = self.rate()
        for p in self.base_optimizer.param_groups:
            p["lr"] = self._rate
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def rate(self, step = None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.scale_factor * self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

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

        self.model_type = cfg['type']

        self.embedding_dim_sqrt = math.sqrt(cfg['embedding-dim'])

        self.pos_encoder = PositionalEncoding(cfg['embedding-dim'], cfg['dropout'])
        encoder_layers = nn.TransformerEncoderLayer(
            cfg['embedding-dim'], 
            cfg['attn-heads'],
            cfg['attn-hidden'],
            cfg['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, cfg['num-layers'])

        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['embedding-dim'], cfg['cls-hidden-dim']),
            nn.Dropout(cfg['dropout']),
            nn.ReLU(),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        if self.cfg['embedding-type'] == 'hash':
            self.embedding.init_weight_range(init_range)
        self.classifier[1].bias.data.zero_()
        self.classifier[4].bias.data.zero_()
        self.classifier[1].weight.data.uniform_(-init_range, init_range)
        self.classifier[4].weight.data.uniform_(-init_range, init_range)

    def forward(self, x, _):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        #x = x.mean(dim=0)
        x = x[0,:,:]
        x = self.classifier(x)
        return x

def run_badboy(model, dl, device, criterion, args):
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = WarmupOptimizer(
        base_optimizer, 
        d_model=model.cfg['embedding-dim'], 
        scale_factor=1, 
        warmup_steps=100
    )
    optimizer = base_optimizer

    def train():
        model.train()
        running_loss, running_corrects, num_examples = 0, 0, 0
        iterator = tqdm(dl['train'], leave=False) if args.loadbar else dl['train']
        for x1, _, target_labels, target_logits in iterator:
            x1 = x1.to(device)
            target_labels = target_labels.to(device).squeeze()
            target_logits = target_logits.to(device)

            optimizer.zero_grad()
            torch.set_grad_enabled(True)

            out = model(x1, None)
            _, preds = torch.max(out, 1)

            loss = criterion(out, target_logits, target_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_corrects += torch.sum(preds == target_labels.data).item()
            num_examples += len(x1)
            running_loss += loss.item() * len(x1)
        accuracy = 0 if num_examples == 0 else running_corrects / num_examples
        print(f'| --> Train loss:     {running_loss/num_examples:.4f}')
        print(f'| --> Train accuracy: {accuracy:.4f}')


    def eval():
        running_corrects, num_examples = 0, 0
        iterator = tqdm(dl['val'], leave=False) if args.loadbar else dl['val']
        with torch.no_grad():
            for x1, _, target_labels,_ in iterator:
                x1 = x1.to(device)
                target_labels = target_labels.to(device).squeeze()
                out = model(x1, None)

                _, preds = torch.max(out, 1)

                running_corrects += torch.sum(preds == target_labels.data).item()
                num_examples += len(x1)
            accuracy = 0 if num_examples == 0 else running_corrects / num_examples
            print(f'| --> Val accuracy:   {accuracy:.4f}') 
    
    for i in range(1, 210):
        print(f'* EPOCH {i}')
        train()
        eval()
