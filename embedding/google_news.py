import torch
import torch.nn as nn

from gensim.models import KeyedVectors

from embedding.base import Embedding

class GoogleNewsEmb(Embedding):
    def __init__(self, cfg, load=True):
        super().__init__(cfg, load)

    def init_embeddings(self):
        wv = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
        vocab = list(wv.key_to_index.keys())[:self.cfg['vocab-size']]

        vectors = []
        self.key_to_index = {}
        for i, key in enumerate(vocab):
            vectors.append(torch.from_numpy(wv[key]))
            self.key_to_index[key] = i

        # append pad and unknown
        vectors.append(torch.FloatTensor(300).uniform_(-0.25, 0.25))
        vectors.append(torch.FloatTensor(300).uniform_(-0.25, 0.25))
        self.pad_idx = self.cfg['vocab-size']
        self.unknown_idx = self.cfg['vocab-size'] + 1
        self.key_to_index['<pad>'] = self.pad_idx
        self.key_to_index['<unknown>'] = self.unknown_idx

        return nn.Embedding.from_pretrained(torch.stack(vectors))

    def encode(self, sent):
        out = []
        for word in sent.split(' '):
            idx = self.key_to_index.get(word, self.key_to_index['<unknown>'])
            out.append(idx)
        return out