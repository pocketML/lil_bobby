import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from bpemb import BPEmb
import torch.nn.functional as F
from common.task_utils import TASK_LABEL_DICT

# mix implemention of https://arxiv.org/pdf/1903.12136.pdf
#  but with bytepair embeddings instead of the humongously
#  sized word2vec GoogleNews pre-trained word embeddings yay
class TangBILSTM(nn.Module):
    def __init__(self, task, use_gpu=True, use_sentence_pairs=False):
        super().__init__()
        self.label_dict = TASK_LABEL_DICT[task]
        self.use_sentence_pairs = use_sentence_pairs
        self.batch_size = 50
        self.embedding_size = 25
        self.vocab_size = 50000
        self.num_hidden_features = 150
        self.num_layers = 1
        self.use_gpu = use_gpu
        self.out_features = 200
        self.num_classes = 2

        self.bpe = BPEmb(lang="en", dim=self.embedding_size, vs=self.vocab_size, add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))

        self.bilstm = nn.LSTM(
            batch_first=True,
            input_size=self.embedding_size,
            hidden_size=self.num_hidden_features,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        if not use_sentence_pairs:
            self.fc1 = nn.Linear(2 * self.num_hidden_features, self.out_features)
        else:
            self.fc1 = nn.Linear(8 * self.num_hidden_features, self.out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.out_features, self.num_classes)

    def init_hidden(self):
        h = torch.zeros(2, self.batch_size, self.num_hidden_features)
        c = torch.zeros(2, self.batch_size, self.num_hidden_features)
        if self.use_gpu:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    def forward(self, x, lens):
        if not self.use_sentence_pairs:
            emb = self.embedding(x).float()
            packed = pack_padded_sequence(emb, lens, batch_first=True)
            out, _ = self.bilstm(packed, self.init_hidden())
            unpacked, _ = pad_packed_sequence(out, batch_first=True)

            len_idx = torch.LongTensor(lens) - 1
            batch_idx = torch.LongTensor([i for i in range(self.batch_size)])
            transmorgyfied = unpacked[batch_idx, len_idx, :]            
        else:
            emb1, emb2 = self.embedding(x[0]), self.embedding(x[1])            
            packed1 = pack_padded_sequence(emb1, lens[0], batch_first=True)
            packed2 = pack_padded_sequence(emb2, lens[1], batch_first=True)
            out1, _ = self.bilstm(packed1, self.init_hidden())
            out2, _ = self.bilstm(packed2, self.init_hidden())
            unpacked1, _ = pad_packed_sequence(out1, batch_first=True)
            unpacked2, _ = pad_packed_sequence(out2, batch_first=True)
            
            len_idx1 = torch.LongTensor(lens[0]) - 1
            len_idx2 = torch.LongTensor(lens[1]) - 1
            batch_idx = torch.LongTensor([i for i in range(self.batch_size)])

            h_n1 = unpacked1[batch_idx, len_idx1, :]
            h_n2 = unpacked2[batch_idx, len_idx2, :]

            transmorgyfied = torch.cat([h_n1, h_n2, torch.abs(h_n1 - h_n2), h_n1 * h_n2], 1)
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
