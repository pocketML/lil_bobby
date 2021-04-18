import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

import unicodedata
import string
import hashlib

from torch.quantization.stubs import DeQuantStub

from compression.distillation.student_models import base

# one-hot encoding of chars
# what would happen if the tensor is for a word
# with the categories being the char count? 
class CharEmbedding():
    def __init__(self):
        #self.vocab = set([x.strip() for x in string.printable if x not in string.ascii_uppercase])
        self.vocab = [x for x in string.ascii_lowercase + " ,.;:?!"]
        self.vocab_size = len(self.vocab)
        self.mapping = {c: i for i, c in enumerate(self.vocab)}
        self.vectors = self.create_vectors()

    # sent to tensor
    # sent_length x 1 x vocab_size
    def create_vectors(self):
        stack = []
        for i in range(self.vocab_size + 1):
            tensor = torch.zeros(self.vocab_size + 1)
            tensor[i] = 1
            stack.append(tensor)
        return torch.stack(stack)

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

# let's keep hash weights to a list of len 256 + 1
class HashEmbedding(nn.Module):
    def __init__(self, num_hashes, embedding_dim):
        super().__init__()
        self.num_hashes = num_hashes
        self.embedding_dim = embedding_dim
        self.K = 512
        self.B = 256
        self.right_shift = 7
        self.vocab_size = self.K
        self.weights = nn.Embedding(self.vocab_size * num_hashes + num_hashes, 1)
        self.embedding = nn.EmbeddingBag(self.B + 1, self.embedding_dim, mode='sum')
        self.is_quantized = False
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def encode(self, sent):
        sent_stack = []
        for word in sent.split(" "):
            word_stack = []
            for i in range(self.num_hashes):
                salted_word =  f'{i}{word}'
                hashed = hashlib.md5(salted_word.encode('utf-8')).digest()[-2:]
                hashed_int = int.from_bytes(hashed, 'big') >> self.right_shift
                word_stack.append(hashed_int)
            sent_stack.append(torch.LongTensor(word_stack))
        out = torch.stack(sent_stack)
        return out

    def prepare_quantization(self):
        self.weights.qconfig = quant.float_qparams_weight_only_qconfig
        self.weights = quantized.Embedding.from_float(self.weights)
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.EmbeddingBag.from_float(self.embedding)
        self.is_quantized = True

    def forward(self, x):
        indices = x // 2
        weight_idx = x
        for i in range(self.num_hashes):
            weight_idx[:,:,i] += i * (self.K + 1)
        weights = self.weights(weight_idx.view(indices.shape[0], -1))
        weights = weights.view(indices.shape[0], indices.shape[1], -1)
        if self.is_quantized:
            x = torch.stack([
                self.embedding(
                    indices[i,:,:],
                    per_sample_weights=weights[i,:,:])
                for i in range(indices.shape[0])
            ])
        else:
            x = torch.stack([
                self.embedding(indices[i,:,:], per_sample_weights=weights[i,:,:])
                for i in range(indices.shape[0])
            ])
        return x
        

class CharRNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg['embedding-type'] == 'hash':
            self.emb = HashEmbedding(cfg['num-hashes'], cfg['embedding-dim'])
            self.embedding = self.emb
            self.cfg['vocab-size'] = self.emb.vocab_size
        else:
            self.emb = nn.Embedding(cfg['vocab-size'] + 1, cfg["embedding-dim"])
            self.embedding = self.emb
 
        self.n_classes = 2
        #self.rnn = base.get_lstm(cfg)
        self.rnn = nn.RNN(cfg["embedding-dim"], cfg['encoder-hidden-dim'], cfg['num-layers'], batch_first=cfg['batch-first'], dropout=cfg['dropout'])
        self.classifier = nn.Sequential(
            nn.Linear(cfg['encoder-hidden-dim'], cfg['cls-hidden-dim']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['cls-hidden-dim'], self.n_classes)
        )

    def forward(self, sents, lengths):
        emb = self.embedding(sents)
        h = base.pack_rnn_unpack(self.rnn, self.cfg, emb, lengths, emb.shape[0])
        x = base.choose_hidden_state(h, lens=lengths, decision='last')
        x = self.classifier(x)
        return x

    def encode(self, sent):
        return self.emb.encode(sent)