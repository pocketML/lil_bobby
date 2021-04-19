import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized
import hashlib


class HashEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_hashes = cfg['num-hashes']
        self.embedding_dim = cfg['embedding-dim']
        self.K = cfg['vocab-size']
        self.ratio = cfg['hash-ratio']
        self.B = self.K // self.ratio
        self.vocab_size = self.K
        self.weights = nn.Embedding(self.vocab_size * self.num_hashes + self.num_hashes, 1)
        self.embedding = nn.EmbeddingBag(self.B + 1, self.embedding_dim, mode='sum')

    def encode(self, sent):
        sent_stack = []
        for word in sent.split(" "):
            word_stack = []
            for i in range(self.num_hashes):
                salted_word =  f'{i}{word}'
                hashed = hashlib.md5(salted_word.encode('utf-8')).digest()[-2:]
                hashed_int = int.from_bytes(hashed, 'big') % self.K
                word_stack.append(hashed_int)
            sent_stack.append(torch.LongTensor(word_stack))
        out = torch.stack(sent_stack)
        return out

    # is inplace
    def prepare_quantization(self):
        self.weights.qconfig = quant.float_qparams_weight_only_qconfig
        self.weights = quantized.Embedding.from_float(self.weights)
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.EmbeddingBag.from_float(self.embedding)
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    def forward(self, x):
        indices = x // self.ratio
        weight_idx = x
        for i in range(self.num_hashes):
            weight_idx[:,:,i] += i * (self.K + 1)
        weights = self.weights(weight_idx.view(indices.shape[0], -1))
        weights = weights.view(indices.shape[0], indices.shape[1], -1)
        x = torch.stack([
            self.embedding(
                indices[i,:,:],
                per_sample_weights=weights[i,:,:])
            for i in range(indices.shape[0])
        ])
        return x