import torch
import torch.nn as nn

from bpemb import BPEmb

from embedding.abstract_class import Embedding

# this is a wrapper for the pre-trained BPEmb by Heinzerling et al.
class BPEmbedding(Embedding):
    def __init__(self, cfg):
        super().__init__()
        self.bpe = BPEmb(lang="en", dim=cfg['embedding-dim'], vs=cfg['vocab-size'], add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))
        self.embedding = self.embedding.float()

    def encode(self, sent):
        return self.bpe.encode_ids(sent)
