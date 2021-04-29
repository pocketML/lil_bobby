import torch.nn as nn

import string
import unicodedata

from embedding.base import Embedding

class CharEmbedding(Embedding):
    def __init__(self, cfg, load=True):
        self.vocab = None
        super().__init__(cfg, load)

    def init_embeddings(self):
        self.vocab = [x for x in string.ascii_lowercase + " ,.;:?!"]
        self.vocab_size = len(self.vocab)
        self.cfg['vocab-size'] = self.vocab_size
        self.mapping = {c: i for i, c in enumerate(self.vocab)}
        return nn.Embedding(self.vocab_size + 1, self.cfg['embedding-dim']) # +1 for pad

    # from https://stackoverflow.com/a/518232      
    def strip_sentence(self, sentence):
        sentence = sentence.lower()
        ascii = ''.join(c for c in unicodedata.normalize('NFD', sentence) 
            if unicodedata.category(c) != 'Mn').encode('ascii', 'ignore').decode('utf-8')
        return ''.join(c for c in ascii if c in self.vocab)

    def encode(self, sent):
        sent = self.strip_sentence(sent).strip()
        if len(sent) <= 0:
            return [self.cfg["vocab-size"]]
        return [self.mapping[c] for c in sent]
