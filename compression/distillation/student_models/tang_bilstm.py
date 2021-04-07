import torch
import torch.nn as nn
from bpemb import BPEmb
from compression.distillation.student_models import base

# mix implemention of https://arxiv.org/pdf/1903.12136.pdf
#  but with bytepair embeddings instead of the humongously
#  sized word2vec GoogleNews pre-trained word embeddings yay
class TangBILSTM(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

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
            nn.ReLU(), 
            nn.Linear(self.cfg['cls-hidden-dim'], self.cfg['num-classes']))

    def forward(self, x, lens):
        def embed_encode_sents(sents, lengths):
            #embedding
            emb = self.embedding(sents).float()
            # encoding
            h = base.pack_bilstm_unpack(self.bilstm, self.cfg, emb, lengths, emb.shape[0])
            return base.choose_hidden_state(h, lens=lengths, decision='last')

        if not self.cfg['use-sentence-pairs']:
            x = embed_encode_sents(x, lens)
        else:
            x1 = embed_encode_sents(x[0], lens[0])
            x2 = embed_encode_sents(x[1], lens[1])
            x = base.cat_cmp(x1, x2)
        # classification
        return self.classifier(x)
