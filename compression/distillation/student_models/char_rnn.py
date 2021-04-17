import unicodedata
import torch
from torch.nn.modules import dropout
from compression.distillation.student_models import base
import string
import torch.nn as nn

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

class CharRNN(base.StudentModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.char_emb = CharEmbedding()
        vocab_size = self.char_emb.vocab_size
        cfg['vocab-size'] = vocab_size
        #cfg['type'] = 'lstm'
        self.embedding = nn.Embedding(vocab_size + 1, cfg["embedding-dim"])#.from_pretrained(self.char_emb.vectors)
 
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
        sent = self.char_emb.strip_sentence(sent).strip()
        if len(sent) <= 0:
            return [self.char_emb.vocab_size]
        return [self.char_emb.mapping[c] for c in sent]
