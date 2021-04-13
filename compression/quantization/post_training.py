import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized
import torch
import copy

import evaluate
from common import data_utils

def quantize_embeddings(model, args, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    val_data = data_utils.load_val_data(args.task)
    dl = data_utils.get_dataloader_dict_val(model, val_data)
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    model.embedding.qconfig = quant.float_qparams_weight_only_qconfig
    model.embedding = nn.Sequential(quantized.Embedding.from_float(model.embedding), quant.DeQuantStub())
    quant.prepare(model.embedding, inplace=True)
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    quant.convert(model.embedding, inplace=True)
    return model

def quantize_encoder(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    model = quant.quantize_dynamic(
        model, qconfig_spec={nn.LSTM}, dtype=torch.qint8
    )
    return model

def quantize_classifier(model, args, type='static', inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    if type == 'static':
        val_data = data_utils.load_val_data(args.task)
        dl = data_utils.get_dataloader_dict_val(model, val_data)
        device = torch.device('cpu') if args.cpu else torch.device('cuda')
        model.classifier = quant.QuantWrapper(model.classifier)
        model.classifier.qconfig = quant.default_qconfig
        quant.prepare(model.classifier, inplace=True)
        evaluate.evaluate_distilled_model(model, dl, device, args, None)
        quant.convert(model.classifier, inplace=True)
    else:
        model.qconfig = quant.default_dynamic_qconfig
        model = quant.quantize_dynamic(
            model, qconfig_spec={nn.Linear}, dtype=torch.qint8
        )
    return model