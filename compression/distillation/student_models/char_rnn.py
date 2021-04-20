import torch
import torch.nn as nn

from embedding import embeddings
from compression.distillation.student_models import base

class CharRNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)
 
        self.n_classes = 2
        self.encoder = nn.RNN(cfg["embedding-dim"], cfg['encoder-hidden-dim'], cfg['num-layers'], batch_first=cfg['batch-first'], dropout=cfg['dropout'])
        self.classifier = nn.Sequential(
            nn.Linear(cfg['encoder-hidden-dim'], cfg['cls-hidden-dim']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['cls-hidden-dim'], self.n_classes)
        )

    def forward(self, sents, lengths):
        emb = self.embedding(sents)
        h = base.pack_rnn_unpack(self.encoder, self.cfg, emb, lengths, emb.shape[0])
        x = base.choose_hidden_state(h, lens=lengths, decision='last')
        x = self.classifier(x)
        return x
