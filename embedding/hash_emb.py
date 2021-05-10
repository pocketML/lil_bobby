import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

import hashlib

from embedding.base import Embedding
from compression.distillation.student_models import base

class HashEmbedding(Embedding):
    def __init__(self, cfg, load=False):
        super().__init__(cfg, load)
        self.num_hashes = cfg['num-hashes']
        self.embedding_dim = cfg['embedding-dim']
        self.K = cfg['vocab-size']
        self.ratio = cfg['hash-ratio']
        self.B = self.K // self.ratio
        self.vocab_size = self.K
        scalar_size = self.vocab_size * self.num_hashes + self.num_hashes
        self.scalars = nn.Embedding(scalar_size, 1)
        self.vectors = nn.EmbeddingBag(self.B + 1, self.embedding_dim, mode='sum')
        self.hash_offsets = torch.LongTensor([i * (self.K + 1) for i in range(self.num_hashes)])

    def encode(self, sent):
        sent_stack = []
        for word in sent.split(" "):
            word_stack = []
            for i in range(self.num_hashes):
                salted_word =  f'{i}{word}'
                hashed = hashlib.md5(salted_word.encode('utf-8')).digest()[-4:]
                hashed_int = int.from_bytes(hashed, 'big') % self.K
                word_stack.append(hashed_int)
            sent_stack.append(torch.LongTensor(word_stack))
        out = torch.stack(sent_stack)
        return out

    # is inplace
    def prepare_to_quantize(self):
        #self.scalars.qconfig = quant.float_qparams_weight_only_qconfig
        #self.scalars = quantized.Embedding.from_float(self.scalars)
        self.vectors.qconfig = quant.float_qparams_weight_only_qconfig
        self.vectors = quantized.EmbeddingBag.from_float(self.vectors)
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        quant.prepare(self.vectors, inplace=True)
        #quant.prepare(self.scalars, inplace=True)

    # is inplace
    def convert_to_quantized(self):
        quant.convert(self.vectors, inplace=True)
        #quant.convert(self.scalars, inplace=True)

    def _apply(self, fn):
        super()._apply(fn)
        self.hash_offsets = fn(self.hash_offsets)
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        scalar_idx = (x + self.hash_offsets).view(batch_size, -1)
        scalars = self.scalars(scalar_idx).view(batch_size * seq_len, -1)
        indices = (x // self.ratio).view(batch_size * seq_len, -1)
        x = self.vectors(indices, per_sample_weights=scalars)
        x = x.view(batch_size, seq_len, -1)
        return x

    def init_weight_range(self, init_range):
        self.scalars.weight.data.fill_(1) #uniform_(-init_range, init_range)
        self.vectors.weight.data.uniform_(-init_range, init_range)