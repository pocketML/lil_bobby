import torch
import torch.nn as nn

from embedding import embeddings
from compression.distillation.student_models import base

class RNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)
        
        self.encoder = base.get_rnn(cfg)

        inp_d = self.cfg['encoder-hidden-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['encoder-hidden-dim']
        inp_d = inp_d * 2 if self.cfg['bidirectional'] else inp_d        
        self.classifier = base.get_classifier(inp_d, cfg)
        
    def forward(self, x, lens):
        def embed_encode_sents(sents, lengths, enforce_sorted=True):
            #embedding
            sents = sents.contiguous()
            emb = self.embedding(sents)
            # encoding
            h = base.pack_rnn_unpack(self.encoder, self.cfg, emb, lengths, emb.shape[0], enforce_sorted=enforce_sorted)
            return base.choose_hidden_state(h, lens=lengths, decision='last')

        if not self.cfg['use-sentence-pairs']:
            x = embed_encode_sents(x, lens)
        else:
            x1 = embed_encode_sents(x[0], lens[0], enforce_sorted=False)
            x2 = embed_encode_sents(x[1], lens[1], enforce_sorted=False)
            x = base.cat_cmp(x1, x2)

        x = self.classifier(x)
        return x