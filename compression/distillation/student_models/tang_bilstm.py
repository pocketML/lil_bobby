import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from bpemb import BPEmb
import torch.nn.functional as F

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

def get_bpemb(lang="en", dim=25, vocab=50000):
    return BPEmb(lang=lang, dim=dim, vs=vocab)

def get_hidden_zeros(use_gpu, batch_size, num_hidden_nodes):
    h = torch.zeros(2, batch_size, num_hidden_nodes)
    c = torch.zeros(2, batch_size, num_hidden_nodes)
    if use_gpu:
        h.cuda()
        c.cuda()
    return (h, c)

# mix implemention of https://arxiv.org/pdf/1903.12136.pdf
#  but with bytepair embeddings instead of the humongously
#  sized word2vec GoogleNews pre-trained word embeddings yay
class TangBILSTM(nn.Module):
    def __init__(self, label_dictionary):
        super().__init__()

        self.label_dict = {label_dictionary.symbols[i]: torch.Tensor([i - 4]) for i in range(len(label_dictionary.symbols))}
        self.embedding_size = 25
        self.num_hidden_features = 150
        self.num_layers = 1
        self.gpu = True
        self.out_features = 200
        self.num_classes = 2
        torch.manual_seed(1)

        self.bpe = get_bpemb(dim=self.embedding_size)
        self.bilstm = nn.LSTM(
            batch_first=True,
            input_size=self.embedding_size,
            hidden_size=self.num_hidden_features,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(2 * self.num_hidden_features, self.out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.out_features, self.num_classes)

    def encode(self, sentences):
        if not isinstance(sentences, list):
            raise TypeError("Encode needs a list of strings")
        out = []
        pad_max = 0
        for sent in sentences:
            t = torch.Tensor(self.bpe.embed(sent))
            if pad_max < len(t):
                pad_max = len(t)
            out.append(t)
        for i in range(len(out)):
            pad_amount = pad_max - len(out[i])
            out[i] = F.pad(input=out[i], pad=(0, 0, 0, pad_amount), mode='constant', value=0)
        out = torch.stack(out, dim=0)
        return out

    def forward(self, x):
        batch_size = len(x)
        x, (h_out, _) = self.bilstm(x, get_hidden_zeros(self.gpu, batch_size, self.num_hidden_features))
        transmorgyfied = torch.cat((h_out[0], h_out[1]), 1)
        x = self.fc1(transmorgyfied)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_loss_function(alpha, criterion_distill, criterion_loss):
    beta = 1 - alpha
    def f(predicted_logits, target_logits, predicted_label, target_label):
        distill = beta * criterion_distill(predicted_logits, target_logits)
        loss = alpha * criterion_loss(predicted_label, target_label)
        return distill + loss
    return f