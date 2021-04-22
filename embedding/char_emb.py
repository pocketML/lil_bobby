import torch.nn as nn

import string
import unicodedata

from embedding.abstract_class import Embedding

class CharEmbedding(Embedding):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.vocab = [x for x in string.ascii_lowercase + " ,.;:?!"]
        self.vocab_size = len(self.vocab)
        cfg['vocab-size'] = self.vocab_size
        self.mapping = {c: i for i, c in enumerate(self.vocab)}
        self.embedding = nn.Embedding(self.vocab_size + 1, cfg['embedding-dim']) # +1 for pad

    # from https://stackoverflow.com/a/518232      
    def strip_sentence(self, sentence):
        sentence = sentence.lower()
        ascii = ''.join(c for c in unicodedata.normalize('NFD', sentence) 
            if unicodedata.category(c) != 'Mn').encode('ascii', 'ignore').decode('utf-8')
        return ''.join(c for c in ascii if c in self.vocab)

    def encode(self, sent):
        sent = self.strip_sentence(sent).strip()
        if len(sent) <= 0:
            return [self.vocab_size]
        return [self.mapping[c] for c in sent]
