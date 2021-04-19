import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized

from bpemb import BPEmb

# this is a wrapper for the pre-trained BPEmb by Heinzerling et al.
class BPEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bpe = BPEmb(lang="en", dim=cfg['embedding-dim'], vs=cfg['vocab-size'], add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.bpe.vectors))
        self.embedding = self.embedding.float()

    def encode(self, sent):
        return self.bpe.encode_ids(sent)
    
     # is inplace
    def prepare_quantization(self):
        self.embedding.qconfig = quant.float_qparams_weight_only_qconfig
        self.embedding = quantized.Embedding.from_float(self.embedding)
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    def forward(self, x):
        return self.embedding(x)