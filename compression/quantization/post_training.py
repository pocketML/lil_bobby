import torch.nn as nn
import torch.quantization as quant
import torch
import copy

def quantize_model(model):
    model = copy.deepcopy(model)
    model.eval()
    model = quant.quantize_dynamic(
        model, qconfig_spec={nn.LSTM}, dtype=torch.qint8
    )
    quant.fuse_modules(model, [nn.Linear, nn.ReLU], inplace=True)
    model.embedding.qconfig = quant.float_qparams_dynamic_qconfig
    model.qconfig = quant.default_qconfig
    quant.prepare(model, inplace=True)
    model = quant.convert(model, inplace=True)
    return model