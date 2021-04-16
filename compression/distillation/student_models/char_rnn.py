import unicodedata
import torch
from compression.distillation.student_models import base
import string

ASCII_START = 32
ASCII_END = 126

class CharRNN(base.StudentModel):
    def __init__(cfg):
        super().__init__(cfg)


    def forward():
        pass


    def encode():
        pass

    def _encode_char(c):
        

    # from https://stackoverflow.com/a/518232      
    def _strip_accents(sentence):
        return ''.join(c for c in unicodedata.normalize('NFD', sentence)
            if unicodedata.category(c) != 'Mn')
  
# one-hot encoding of chars
# what would happen if the tensor is for a word
# with the categories being the char count? 
class CharRNNEmbedding():
    def __init__():
        vocab = string.ascii_letters
        vocab += "0123456789,.-;:!\"\'()?&%@="
        self.vocab_size = len(vocab)
        self.mapping = {c: i for i, c in enumerate(vocab)}

    # word to tensor
    # word_length x 1 x vocab_size
    def __call__(self, word):
        tensor = torch.zeros(len(word), 1, self.vocab_size)
        for i, c in enumerate(word):
            tensor[i][0][self.mapping[c]] = 1
