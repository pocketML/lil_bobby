import torch.nn as nn
import torch.quantization as quant
import torch
import copy

def quantize_model(model):
    torch.backends.quantized.engine = 'fbgemm'
    model = copy.deepcopy(model)
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    model.classifier = QuantWrapper(model.classifier)
    quant.prepare_qat(model, inplace=True)
    # quantization aware training goes here
    quant.convert(model.eval(), inplace=True)
    return model