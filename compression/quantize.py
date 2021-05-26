import torch.nn as nn
import torch.quantization as quant
import torch

import copy

import evaluate
from analysis import pretty_print
from common import data_utils

def quantize_embeddings(model, args, dl, device, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    model.embedding.prepare_to_quantize()
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    model.embedding.convert_to_quantized()
    return model

def quantize_encoder(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    model = quant.quantize_dynamic(
        model, qconfig_spec={nn.LSTM}, dtype=torch.qint8
    )
    return model

def quantize_classifier(model, args, dl, device, type='static', inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    if type == 'static':
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

def quantize_model(model, device, args):
    dl = data_utils.get_val_dataloader(model, data_utils.load_val_data(args.task))
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    print("Starting point:")
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    pretty_print.print_model_disk_size(model)
    print()

    if args.ptq_embedding:
        print("** Quantizing embedding layer... **")
        model = quantize_embeddings(model, args, dl, device)
    if args.dq_encoder:
        print("** Quantizing encoder... **")
        model = quantize_encoder(model)
    if args.dq_classifier:
        print("** Quantizing classifier... **")
        model = quantize_classifier(model, args, dl, device, type='dynamic')
    elif args.ptq_classifier:
        print("** Quantizing classifier... **")
        model = quantize_classifier(model, args, dl, device, type='static')

    print('** Quantization completed **')
    pretty_print.print_model_disk_size(model)
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    print()
    return model