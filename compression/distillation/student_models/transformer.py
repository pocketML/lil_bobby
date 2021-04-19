import math
from bpemb import BPEmb
from compression.distillation.student_models import base
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_encode = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pos_encode[:, 0::2] = torch.sin(position * div_term)
        self.pos_encode[:, 1::2] = torch.cos(position * div_term)
        self.pos_encode = self.pos_encode.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pos_encode[:x.size(0), :]
        return self.dropout(x)

class Transformer(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.max_seq_len = 150
        self.bpe = BPEmb(lang="en", dim=self.cfg['embedding-dim'], vs=self.cfg['vocab-size'], add_pad_emb=True)
        self.pos_encoder = PositionalEncoding(cfg["embedding-dim"], cfg["dropout"])
        if cfg["use-gpu"]:
            self.pos_encoder.pos_encode = self.pos_encoder.pos_encode.cuda()
        self.transformer_encoder = torch.nn.TransformerEncoderLayer(
            cfg["embedding-dim"], cfg["attn-heads"], cfg["encoder-hidden-dim"], cfg["dropout"]
        )
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors)).float()
        self.decoder = nn.Linear(cfg["embedding-dim"], cfg["vocab-size"])
        self.classifier_dim = self.max_seq_len * cfg["vocab-size"]
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, cfg["cls-hidden-dim"]),
            nn.ReLU(), 
            nn.Linear(cfg["cls-hidden-dim"], cfg["num-classes"])
        )

    def forward(self, x, lens):
        emb = self.embedding(x) * math.sqrt(self.cfg["embedding-dim"])
        pad_amount = self.max_seq_len - emb.shape[1]

        # Pad 2nd dimension to match max_seq_len.
        emb = torch.nn.functional.pad(emb, pad=(0, 0, 0, pad_amount, 0, 0), mode="constant")

        x = self.pos_encoder(emb)
        x = self.transformer_encoder(x)
        x = self.decoder(x)

        x = x.view(-1, self.classifier_dim)

        x = self.classifier(x)
        return x
