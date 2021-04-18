import compression.distillation.student_models.char_rnn as c
import torch
import torch.nn as nn
import torch.quantization as quant

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(2,10)
        self.qemb = nn.quantized.EmbeddingBag(2, 10)

    def forward(self, x, use_quantized):
        if use_quantized:
            return self.qemb(x)
        return self.emb(x)
backend = 'fbgemm'
torch.backends.quantized.engine = backend
#torch.manual_seed(1)
model = Model()
#model.qemb.qconfig = quant.float_qparams_weight_only_qconfig
#model.qconfig = quant.default_qconfig
#quant.prepare(model.qemb, inplace=True)

input1 = torch.stack([torch.LongTensor([0])])
input2 = torch.stack([torch.LongTensor([0,1])])
out1 = model(input1, False)
print(out1, out1.dtype)
qout1 = model(input1, True)
print(qout1, qout1.dtype)
out2 = model(input2, False)
print(out2, out2.dtype)
qout2 = model(input2, True)
print(qout2, qout2.dtype)

#quant.convert(model.qemb, inplace=True)
