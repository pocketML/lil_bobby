import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

import string
import unicodedata

class CharEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.vocab = [x for x in string.ascii_lowercase + " ,.;:?!"]
        self.vocab_size = len(self.vocab)
        cfg['vocab-size'] = self.vocab_size
        self.mapping = {c: i for i, c in enumerate(self.vocab)}
        self.embedding = nn.Embedding(self.vocab_size, cfg['embedding-dim'])

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
        
    # is inplace
    def prepare_quantization(self):
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.Embedding.from_float(self.embedding)
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    def forward(self, x):
        return self.embedding(x)