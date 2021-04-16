import unicodedata
import torch
from compression.distillation.student_models import base
import string
import torch.nn as nn

# one-hot encoding of chars
# what would happen if the tensor is for a word
# with the categories being the char count? 
class CharEmbedding():
    def __init__(self):
        self.vocab = set([x.strip() for x in string.printable if x not in string.ascii_uppercase])
        self.vocab.add(" ")
        self.vocab.remove("")
        self.vocab_size = len(self.vocab) + 1
        self.mapping = {c: i for i, c in enumerate(self.vocab)}

    # sent to tensor
    # sent_length x 1 x vocab_size
    def __call__(self, batch):
        batch_stack = []
        for idx in batch:
            sent_stack = []
            for id in idx:
                tensor = torch.zeros(self.vocab_size)
                tensor[id] = 1
                sent_stack.append(tensor)
            batch_stack.append(sent_stack)
        return torch.stack(batch_stack)

    # from https://stackoverflow.com/a/518232      
    def strip_accents(self, sentence):
        return ''.join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn').encode('ascii', 'ignore').decode('utf-8')

class CharRNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.hidden_size= 128
        self.embedding = CharEmbedding()
        cfg['vocab-size'] = self.embedding.vocab_size
 
        self.input_size = self.embedding.vocab_size
        self.n_classes = 2
        self.h = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.o = nn.Linear(self.hidden_size + self.input_size, self.n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sents, lengths):
        batch_size = len(sents)
        hidden = self.init_hidden(batch_size)
        out = torch.empty((batch_size, 1, self.n_classes))
        emb = self.embedding(sents)
        for i in range(emb.shape[1]):
            o, h = self.forward_char(emb[:,i,:], hidden[:,i,:])
            hidden = torch.cat((hidden, h), 1)
            out = torch.cat((out, o), 1)
        batch_idx = torch.LongTensor([i for i in range(batch_size)]) # 1 = batch_size
        lens = torch.LongTensor(lengths)
        hidden = hidden[batch_idx, lens - 1, :]
        out = out[batch_idx, lens - 1, :]
        return out, hidden

    def forward_char(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.h(combined)
        out = self.o(combined)
        out = self.softmax(out)
        return out.view(x.shape[0],1,-1), hidden.view(x.shape[0],1,-1)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, 1, self.hidden_size)

    def encode(self, sent):
        sent = self.strip_accents(sent).strip().lower()
        [self.embedding.mapping[c] for c in sent]

