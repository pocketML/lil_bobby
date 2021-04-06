import torch
import torch.nn as nn
from bpemb import BPEmb
from common.task_utils import TASK_LABEL_DICT
from compression.distillation.student_models import base

# Model inspired by https://openreview.net/pdf?id=rJ4km2R5t7
# https://github.com/nyu-mll/GLUE-baselines/tree/master/src
class GlueBILSTM(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dropout = nn.Dropout(p=self.cfg['dropout']) if self.cfg['dropout'] else lambda x: x

        # embedding
        self.bpe = BPEmb(lang="en", dim=self.cfg['embedding-dim'], vs=self.cfg['vocab-size'], add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))
        
        # encoding
        self.bilstm = base.get_lstm(self.cfg)

        # classifier/mlp
        inp_d = self.cfg['encoder-hidden-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['encoder-hidden-dim']
        inp_d = inp_d * 2 if self.cfg['bidirectional'] else inp_d
        self.classifier = nn.Sequential(
            nn.Linear(inp_d, self.cfg['cls-hidden-dim']),
            nn.Tanh(), 
            nn.Linear(self.cfg['cls-hidden-dim'], self.cfg['num-classes']))

    def forward(self, x, lens):
        def embed_enc_sents(sents, lengths):
            #embedding
            emb = self.embedding(sents).float()
            emb = self.dropout(emb)
            # encoding
            h = base.pack_bilstm_unpack(self.bilstm, self.cfg, emb, lengths)
            h = self.dropout(h)
            return base.choose_hidden_state(h, decision='max')
        
        if not self.cfg['use-sentence-pairs']: 
            x = embed_enc_sents(x, lens)
        else:
            x1 = embed_enc_sents(x[0], lens[0])
            x2 = embed_enc_sents(x[1], lens[1])
            x = base.cat_cmp(x1, x2)
        # classifier
        x = self.classifier(x)
        return x
