import torch.nn as nn

from common.task_utils import TASK_LABEL_DICT
from compression.distillation.student_models import base
from embedding import embeddings

# Model inspired by https://openreview.net/pdf?id=rJ4km2R5t7
# https://github.com/nyu-mll/GLUE-baselines/tree/master/src
class GlueBILSTM(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dropout = nn.Dropout(p=self.cfg['dropout']) if self.cfg['dropout'] else lambda x: x

        # embedding
        self.embedding = embeddings.get_embedding(cfg)
        
        # encoding
        self.bilstm = base.get_lstm(self.cfg)

        # classifier/mlp
        inp_d = self.cfg['encoder-hidden-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['encoder-hidden-dim']
        inp_d = inp_d * 2 if self.cfg['bidirectional'] else inp_d
        self.classifier = nn.Sequential(
            nn.Linear(inp_d, self.cfg['cls-hidden-dim']),
            nn.Tanh(), 
            nn.Linear(self.cfg['cls-hidden-dim'], self.cfg['num-classes'])
        )

    def forward(self, x, lens):
        def embed_enc_sents(sents, lengths):
            #embedding
            sents = sents.contiguous()
            emb = self.embedding(sents)
            emb = self.dropout(emb)
            # encoding
            h = base.pack_rnn_unpack(self.bilstm, self.cfg, emb, lengths, emb.shape[0])
            h = self.dropout(h)
            return base.choose_hidden_state(h, decision='max')
        
        if not self.cfg['use-sentence-pairs']: 
            x = embed_enc_sents(x, lens)
        else:
            x1 = embed_enc_sents(x[0], lens[0])
            x2 = embed_enc_sents(x[1], lens[1])
            x = base.cat_cmp(x1, x2)
        #print("whelp6")
        # classifier
        x = self.classifier(x)
        return x
