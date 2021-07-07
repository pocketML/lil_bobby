import torch.nn as nn
import torch.quantization as quant
import torch

import copy

import evaluate
from analysis import pretty_print
from common import data_utils
from compression.distillation.student_models.rnn import QuantizableRNN

def quantize_embeddings(model, args, dl, device, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.embedding.prepare_to_quantize()
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    model.embedding.convert_to_quantized()
    return model

def quantize_encoder(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    if model.cfg['type'] == 'lstm':
        model.bilstm = quant.quantize_dynamic(
        model.bilstm, qconfig_spec={nn.LSTM}, dtype=torch.qint8
    )
    elif model.cfg['type'] == 'rnn':
        model.encoder = QuantizableRNN(model.encoder, model.cfg)
        model.encoder = quant.quantize_dynamic(
            model.encoder, qconfig_spec={nn.Linear}, dtype=torch.qint8
        )
    return model

def dynamic_quantize_rnn_and_classifier(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    model.encoder = QuantizableRNN(model.encoder, model.cfg)
    model.qconfig = quant.default_dynamic_qconfig
    model = quant.quantize_dynamic(
        model, qconfig_spec={nn.Linear}, dtype=torch.qint8
    )
    return model

def quantize_classifier(model, args, dl, device, type='static', inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    if type == 'static':
        model.classifier = quant.QuantWrapper(model.classifier)
        model.classifier.qconfig = quant.default_qconfig
        quant.prepare(model.classifier, inplace=True)
        evaluate.evaluate_distilled_model(model, dl, device, args, None)
        quant.convert(model.classifier, inplace=True)
    elif type == 'dynamic':
        model.qconfig = quant.default_dynamic_qconfig
        model.classifier = quant.quantize_dynamic(
            model.classifier, qconfig_spec={nn.Linear}, dtype=torch.qint8
        )
    else:
        raise Exception(f'Quantization method {type} not supported')
    return model

def quantize_model(model, device, args, sacred_experiment=None):
    dl = data_utils.get_val_dataloader(model, data_utils.load_val_data(args.task))
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    print("Starting point:")
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    pretty_print.print_model_disk_size(model, sacred_experiment)
    print()
    model.eval()
    if args.ptq_embedding:
        print("** Quantizing embedding layer... **")
        model = quantize_embeddings(model, args, dl, device)
    if model.cfg['type'] == 'rnn' and args.dq_encoder and args.dq_classifier:
        # if quant of both rnn and classifier we have to do it together
        print("** Quantizing encoder... **")
        print("** Quantizing classifier... **")
        model = dynamic_quantize_rnn_and_classifier(model)
    elif args.dq_encoder:
        print("** Quantizing encoder... **")
        model = quantize_encoder(model)
    if not (model.cfg['type'] == 'rnn' and args.dq_encoder):
        # for some reason the model becomes broken if 
        if args.dq_classifier and not (model.cfg['type'] == 'rnn' and args.dq_encoder):
            print("** Quantizing classifier... **")
            model = quantize_classifier(model, args, dl, device, type='dynamic')
        if args.ptq_classifier:
            print("** Quantizing classifier... **")
            model = quantize_classifier(model, args, dl, device, type='static')

    print('** Quantization completed **')
    pretty_print.print_model_disk_size(model, sacred_experiment)
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    print()
    return model