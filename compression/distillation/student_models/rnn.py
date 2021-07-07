import torch
import torch.nn as nn
import torch.quantization as quant
from copy import deepcopy

from embedding import embeddings
from compression.distillation.student_models import base

class QuantizableRNN(nn.Module):
    def __init__(self, rnn, cfg):
        super().__init__()

        self.input_size = cfg['embedding-dim']
        self.hidden_size = cfg['encoder-hidden-dim']
        self.bias = True
        self.batch_first = cfg['batch-first']
        self.dropout = cfg['dropout'] > 0.0
        self.bidirectional = cfg['bidirectional']
        self.tanh = nn.Tanh()
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.is_quantized = False

        if isinstance(rnn, QuantizableRNN):
            self.forward_i = deepcopy(rnn.forward_i)
            self.forward_h = deepcopy(rnn.forward_h)
            if self.bidirectional:
                self.reverse_i = deepcopy(rnn.reverse_i)
                self.reverse_h = deepcopy(rnn.reverse_h)
        elif isinstance(rnn, nn.RNN):
            self.forward_i = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
            self.forward_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.forward_i.weight = deepcopy(getattr(rnn, 'weight_ih_l0'))
            self.forward_h.weight = deepcopy(getattr(rnn, 'weight_hh_l0'))
            if self.bias:
                self.forward_i.bias = deepcopy(getattr(rnn, 'bias_ih_l0'))
                self.forward_h.bias = deepcopy(getattr(rnn, 'bias_hh_l0'))
            if self.bidirectional:
                self.reverse_i = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
                self.reverse_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
                self.reverse_i.weight = deepcopy(getattr(rnn, 'weight_ih_l0_reverse'))
                self.reverse_h.weight = deepcopy(getattr(rnn, 'weight_hh_l0_reverse'))
                if self.bias:
                    self.reverse_i.bias = deepcopy(getattr(rnn, 'bias_ih_l0_reverse'))
                    self.reverse_h.bias = deepcopy(getattr(rnn, 'bias_hh_l0_reverse'))
        else:
            raise Exception(f'Couldn\'t create QuantizableRNN from type {type(rnn)}')

    def forward(self, sents, lens):
        max_seq_len = len(sents[0])
        batch_size = len(sents)
        out = torch.empty(batch_size, max_seq_len, self.hidden_size + self.hidden_size * int(self.bidirectional))
        hx = torch.zeros(sents.shape[0], self.hidden_size)

        if self.bidirectional:
            sents_rev = torch.empty(batch_size, max_seq_len, self.input_size)
            for i, length in enumerate(lens):
                sents_rev[i,:length] = sents[i,:length]
                ''' 
                    Is there a bug in our regular RNN implementation
                    since the above and the not line below is necessary
                    to get a matching evaluation score?
                    I.e. is our bidirectionality simply two "different" RNNs,
                    going in the same LR direction that concatenates their results?
                '''
                #sents_rev[i,:length] = torch.fliplr(sents[i,:length])
            hx_rev = torch.zeros(sents.shape[0], self.hidden_size)
        for i in range(max_seq_len):
            hx = self.tanh(self.forward_h(hx) + self.forward_i(sents[:,i,:]))
            if self.bidirectional:
                hx_rev = self.tanh(self.reverse_h(hx_rev) + self.reverse_i(sents_rev[:,i,:]))
                out[:,i,:] = torch.cat((hx, hx_rev), dim=1)
            else:
                out[:,i,:] = hx
        return out

class RNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.embedding = embeddings.get_embedding(cfg)
        
        self.encoder = base.get_rnn(cfg)

        inp_d = self.cfg['encoder-hidden-dim'] * 4 if self.cfg['use-sentence-pairs'] else self.cfg['encoder-hidden-dim']
        inp_d = inp_d * 2 if self.cfg['bidirectional'] else inp_d        
        self.classifier = base.get_classifier(inp_d, cfg)
        
    def forward(self, x, lens):
        def embed_encode_sents(sents, lengths, enforce_sorted=True):
            #embedding
            sents = sents.contiguous()
            emb = self.embedding(sents)
            # encoding
            if isinstance(self.encoder, QuantizableRNN):
                h = self.encoder(emb, lengths)
            else:
                h = base.pack_rnn_unpack(self.encoder, self.cfg, emb, lengths, emb.shape[0], enforce_sorted=enforce_sorted)
            return base.choose_hidden_state(h, lens=lengths, decision='last')

        if not self.cfg['use-sentence-pairs']:
            x = embed_encode_sents(x, lens)
        else:
            x1 = embed_encode_sents(x[0], lens[0], enforce_sorted=False)
            x2 = embed_encode_sents(x[1], lens[1], enforce_sorted=False)
            x = base.cat_cmp(x1, x2)

        x = self.classifier(x)
        return x