import torch
import torch.nn as nn

from embedding import embeddings
from compression.distillation.student_models import base

class CharRNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)
        
        self.encoder = nn.RNN(cfg["embedding-dim"], cfg['encoder-hidden-dim'])

        inp_d = self.cfg['encoder-hidden-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['encoder-hidden-dim']
        inp_d = inp_d * 2 if self.cfg['bidirectional'] else inp_d        
        self.classifier = base.get_classifier(inp_d, cfg)
        self.init_weights(embedding_init_range=0.1, classifier_init_range=0.1)

    def forward(self, sents, lengths):
        emb = self.embedding(sents)
        h = base.pack_rnn_unpack(self.encoder, self.cfg, emb, lengths, emb.shape[0])
        x = base.choose_hidden_state(h, lens=lengths, decision='last')
        x = self.classifier(x)
        return x
