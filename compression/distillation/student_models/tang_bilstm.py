import torch.nn as nn
from torch.optim.adadelta import Adadelta

from embedding import embeddings
from compression.distillation.student_models import base

# mix implemention of https://arxiv.org/pdf/1903.12136.pdf
#  but with bytepair embeddings instead of the humongously
#  sized word2vec GoogleNews pre-trained word embeddings yay
class TangBILSTM(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # embedding
        self.embedding = embeddings.get_embedding(cfg)
        
        # encoding
        self.bilstm = base.get_lstm(self.cfg)

        # classifier/mlp
        inp_d = self.cfg['encoder-hidden-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['encoder-hidden-dim']
        inp_d = inp_d * 2 if self.cfg['bidirectional'] else inp_d
        fc1 = nn.Linear(inp_d, self.cfg['cls-hidden-dim'])
        relu = nn.ReLU()
        dropout = nn.Dropout(self.cfg["dropout"])
        fc2 = nn.Linear(self.cfg['cls-hidden-dim'], self.cfg['num-classes'])
        self.classifier = nn.Sequential(fc1, relu, dropout, fc2)

    def get_optimizer(self):
        return Adadelta(
            self.parameters(), lr=self.cfg['lr'],
            rho=self.cfg['rho']
        )

    def forward(self, x, lens):
        def embed_encode_sents(sents, lengths):
            #embedding
            sents = sents.contiguous()
            emb = self.embedding(sents)
            # encoding
            h = base.pack_rnn_unpack(self.bilstm, self.cfg, emb, lengths, emb.shape[0])
            return base.choose_hidden_state(h, lens=lengths, decision='last')

        if not self.cfg['use-sentence-pairs']:
            x = embed_encode_sents(x, lens)
        else:
            x1 = embed_encode_sents(x[0], lens[0])
            x2 = embed_encode_sents(x[1], lens[1])
            x = base.cat_cmp(x1, x2)
        # classification
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc2(x)
        return self.classifier(x)
