import torch
import torch.nn as nn

import math
from tqdm import tqdm

from compression.distillation.student_models import base
from embedding import embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['cls-hidden-dim'], cfg['num-classes'])
        )
        
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        if self.cfg['embedding-type'] == 'hash':
            self.embedding.init_weight_range(init_range)
        self.classifier[1].bias.data.zero_()
        self.classifier[4].bias.data.zero_()
        self.classifier[1].weight.data.uniform_(-init_range, init_range)
        self.classifier[4].weight.data.uniform_(-init_range, init_range)

    def forward(self, x, x_mask):
        x = self.embedding(x) * self.embedding_dim_sqrt
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, x_mask)
        x = x[:,0,:]
        x = self.classifier(x)
        return x

def run_badboy(model, dl, device, criterion, args):
    lr = 8.0#1.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train():
        model.train()
        running_loss, running_corrects, num_examples = 0, 0, 0
        x_mask = model.generate_square_subsequent_mask(model.cfg['batch-size']).to(device)
        iterator = tqdm(dl['train'], leave=False) if args.loadbar else dl['train']
        for x1, _, target_labels, target_logits in iterator:
            x1 = x1.to(device)
            target_labels = target_labels.to(device).squeeze()
            target_logits = target_logits.to(device)
            if x1.shape[0] != model.cfg['batch-size']:
                x_mask = model.generate_square_subsequent_mask(x1.shape[0]).to(device)

            optimizer.zero_grad()
            torch.set_grad_enabled(True)

            out = model(x1, x_mask)
            #out = torch.F.log_softmax(out, dim=1)
            _, preds = torch.max(out, 1)

            if True:
                print(out)
                #print(target_logits)
                print()

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
        x_mask = model.generate_square_subsequent_mask(model.cfg['batch-size']).to(device)
        iterator = tqdm(dl['val'], leave=False) if args.loadbar else dl['val']
        with torch.no_grad():
            for x1, _, target_labels,_ in iterator:
                x1 = x1.to(device)
                target_labels = target_labels.to(device).squeeze()

                if x1.shape[0] != model.cfg['batch-size']:
                    x_mask = model.generate_square_subsequent_mask(x1.shape[0]).to(device)
                out = model(x1, x_mask)
                out = nn.F.log_softmax(out, dim=1)

                _, preds = torch.max(out, 1)

                running_corrects += torch.sum(preds == target_labels.data).item()
                num_examples += len(x1)
            accuracy = 0 if num_examples == 0 else running_corrects / num_examples
            print(f'| --> Val accuracy:   {accuracy:.4f}') 
    
    for i in range(1, 21):
        print(f'* EPOCH {i}')
        train()
        eval()
        scheduler.step()