import torch
import torch.nn as nn

from bpemb import BPEmb

from embedding.base import Embedding

# this is a wrapper for the pre-trained BPEmb by Heinzerling et al.
class BPEmbedding(Embedding):
    def __init__(self, cfg, load=True):
        super().__init__(cfg, load)

    def init_embeddings(self):
        self.bpe = BPEmb(lang="en", dim=self.cfg['embedding-dim'], vs=self.cfg['vocab-size'], add_pad_emb=True)
        return nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors)).float()

    def encode(self, sent):
        return self.bpe.encode_ids(sent)
