import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

from abc import abstractmethod

class Embedding(nn.Module):
    def __init__(self, cfg, load=True):
        super().__init__()
        self.cfg = cfg
        self.load_pretrained = load
        # Load pre-trained embeddings.
        self.specials = None
        self.mapping = None
        self.embedding = self.init_embeddings()
        if load: # Freeze or unfreeze embeddings if pretrained.
            self.embedding.weight.requires_grad = not cfg["embedding-freeze"]

    def freeze_(self):
        self.embedding.weight.requires_grad = False

    def unfreeze_(self):
        self.embedding.weight.requires_grad = True

    def prepare_to_quantize(self):
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.Embedding.from_float(self.embedding)
        self.qconfig = quant.get_default_qconfig('fbgemm')
        quant.prepare(self.embedding, inplace=True)

    def convert_to_quantized(self):
        quant.convert(self.embedding, inplace=True)

    def forward(self, x):
        return self.embedding(x)

    def init_embeddings(self):
        return None

    def init_weight_range(self, init_range):
        self.embedding.weight.data.uniform_(-init_range, init_range)

    @abstractmethod
    def encode(self, sent):
        pass
