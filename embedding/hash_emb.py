import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

import hashlib

from embedding.base import Embedding
from compression.distillation.student_models import base

class HashEmbedding(Embedding):
    def __init__(self, cfg, load=True):
        super().__init__(cfg, load)
        self.num_hashes = cfg['num-hashes']
        self.embedding_dim = cfg['embedding-dim']
        self.K = cfg['vocab-size']
        self.ratio = cfg['hash-ratio']
        self.B = self.K // self.ratio
        self.vocab_size = self.K
        scalar_size = self.vocab_size * self.num_hashes + self.num_hashes
        self.weights = nn.Embedding(scalar_size, 1)
        self.embedding = nn.EmbeddingBag(self.B + 1, self.embedding_dim, mode='sum')
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
        self.weights.qconfig = quant.float_qparams_weight_only_qconfig
        self.weights.qscheme = torch.per_tensor_affine
        self.weights = quantized.Embedding.from_float(self.weights)
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.EmbeddingBag.from_float(self.embedding)
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        quant.prepare(self.embedding, inplace=True)
        quant.prepare(self.weights, inplace=True)

    # is inplace
    def convert_to_quantized(self):
        quant.convert(self.embedding, inplace=True)
        quant.convert(self.weights, inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        if self.cfg['use-gpu']:
            weight_idx = x + self.hash_offsets.cuda()
        else:
            weight_idx = x + self.hash_offsets
        weights = self.weights(weight_idx.view(batch_size, -1))
        weights = weights.view(batch_size * seq_len, -1)

        indices = (x // self.ratio).view(batch_size * seq_len, -1)
        x = self.embedding(indices, per_sample_weights=weights)
        x = x.view(batch_size, seq_len, -1)

        return x

    def init_weight_range(self, init_range):
        self.weights.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data.uniform_(-init_range, init_range)