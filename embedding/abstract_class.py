import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

from abc import abstractmethod

class Embedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = None

    def prepare_to_quantize(self):
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.Embedding.from_float(self.embedding)
        self.qconfig = quant.get_default_qconfig('fbgemm')
        quant.prepare(self.embedding, inplace=True)

    def convert_to_quantized(self):
        quant.convert(self.embedding, inplace=True)

    def forward(self, x):
        return self.embedding(x)

    @abstractmethod
    def encode(self, sent):
        pass