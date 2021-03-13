import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

# https://arxiv.org/pdf/1903.12136.pdf
class BILSTMConfig:
    def __init__(self):
        self.lr = 1e-4
        self.batch_size = 64
        self.dropout_keep_prob = 0.5
        self.embedding_size = 300
        self.num_classes = 2
        self.num_hidden_nodes = 150
        self.hidden_dim2 = 128
        self.num_layers = 1
        self.bidirectional = True
        self.vocab_size = 50000

class TangBILSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BILSTMConfig()
        self.embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_size)
        self.bilstm = nn.LSTM(
            batch_first=True,
            input_size=self.config.embedding_size,
            hidden_size=self.config.num_hidden_nodes,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
        )
    
    def forward(self, x):
        x, (self.h_0, self.c_0) = self.bilstm(x, (self.h_0, self.c_0))
        last_step = x[-1]


