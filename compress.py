import torch
import torch.nn as nn

from argparse import ArgumentError
import json
import os
import warnings
import random
import numpy as np

from common import argparsers, data_utils, transponder
from compression.distill import train_loop, save_checkpoint
from compression.distillation.models import DistLossFunction, load_student
import evaluate
from compression import prune
from compression import quantize as ptq
from analysis import parameters

warnings.simplefilter("ignore", UserWarning)

def quantize_model(model, device, args):
    dl = data_utils.get_dataloader_dict_val(model, data_utils.load_val_data(args.task))
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    print("Starting point:")
    parameters.print_model_disk_size(model)
    print()

    if args.ptq_embedding:
        print("** quantizing embedding layer **")
        model = ptq.quantize_embeddings(model, args, dl, device)
    if args.dq_encoder:
        print("** quantizing encoder **")
        model = ptq.quantize_encoder(model)
    if args.dq_classifier:
        print("** quantizing classifier **")
        model = ptq.quantize_classifier(model, args, dl, device, type='dynamic')
    elif args.ptq_classifier:
        print("** quantizing classifier **")
        model = ptq.quantize_classifier(model, args, dl, device, type='static')

    print('** Quantization finished.. **')
    parameters.print_model_disk_size(model)
    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    print()
    return model

def do_pruning(model, args, epoch=None):
    threshold = args.prune_threshold
    if epoch is not None:
        threshold = threshold * (epoch / args.prune_warmup)

    if args.prune_magnitude:
        model = prune.magnitude_pruning(model, threshold)
    elif args.prune_movement:
        model = prune.movement_pruning(model, threshold)
    elif args.prune_topk:
        model = prune.topk_pruning(model, threshold)

    return model

def prune_model(model, device, args):
    dl = data_utils.get_dataloader_dict_val(model, data_utils.load_val_data(args.task))

    params, zero = prune.params_zero(model)
    sparsity = (zero / params) * 100
    print(f"Sparsity: {sparsity:.2f}%")

    parameters.print_model_disk_size(model)
    evaluate.evaluate_distilled_model(model, dl, device, args)

    model = do_pruning(model, args)

    params, zero = prune.params_zero(model)
    sparsity = (zero / params) * 100
    print(f"Sparsity: {sparsity:.2f}%")

    parameters.print_model_disk_size(model)
    evaluate.evaluate_distilled_model(model, dl, device, args)
    print()
    return model

def distill_model(task, model, device, args, callback, sacred_experiment):
    temperature = args.temperature
    model.to(device)

    data_splitter = None
    if args.chunk_ratio < 1.0 or args.data_ratio < 1.0:
        data_splitter = data_utils.DataSplitter(
            chunk_ratio=args.chunk_ratio, data_ratio=args.data_ratio
        )

    best_val_acc = 0
    no_improvement = 0
    chunk = 1
    epoch = 1

    distillation_data = None
    dataloader_train = None
    val_data = None

    criterion = DistLossFunction(
        args.alpha, 
        nn.MSELoss(), 
        nn.CrossEntropyLoss(), 
        device,
        temperature=temperature,
    )

    optim = model.get_optimizer()

    val_data = data_utils.load_val_data(task)
    print(f"*** Loaded {len(val_data[0])} validation data samples ***")
    dataloader_val = data_utils.get_dataloader_dict_val(model, val_data, loadbar=args.loadbar)

    while epoch <= args.epochs:
        if distillation_data is None or args.chunk_ratio < 1.0:
            del distillation_data
            distillation_data = data_utils.load_all_distillation_data(
                task, only_original_data=args.original_data, data_splitter=data_splitter
            )
            print(f"*** Loaded {len(distillation_data[0])} training data samples ***")

            print('*** Preparing data for Dataloaders ***')
            dataloader_train = data_utils.get_dataload_dict_train(model, distillation_data, loadbar=args.loadbar)
            print('*** Dataloaders created ***')

        desc = f'* Epoch {epoch}'
        if args.chunk_ratio < 1.0:
            desc += f" - Chunk: {chunk} / {int(1.0 / args.chunk_ratio)}"

        print(desc)

        dataloaders = {"train": dataloader_train, "val": dataloader_val}

        val_acc = train_loop(
            model, criterion, optim, dataloaders, device, args
        )

        chunk += 1
        del dataloaders

        if data_splitter is None or data_splitter == []:
            if val_acc > best_val_acc:
                print(f'Saving new best model')
                best_val_acc = val_acc
                save_checkpoint(model, args.student_arch, sacred_experiment)
                no_improvement = 0
            else:
                no_improvement += 1

            transponder.send_train_status(epoch, val_acc)

            if sacred_experiment is not None:
                sacred_experiment.log_scalar("validation.acc", val_acc)

            if callback is not None:
                model = callback(model, args, epoch)

            if no_improvement == args.early_stopping:
                break

            epoch += 1

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    task = args.task
    student_type = args.student_arch
    seed = args.seed

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    should_prune = "prune" in args.compression_actions
    should_quantize = "quantize" in args.compression_actions

    model = None

    if "distill" in args.compression_actions:
        model = load_student(task, student_type, use_gpu=use_gpu, args=args)
        if sacred_experiment is not None:
            temp_name = f"temp_{sacred_experiment.info['name']}.json"
            with open(temp_name, "w", encoding="utf-8") as fp:
                json.dump(model.cfg, fp, indent=4)
            sacred_experiment.add_artifact(temp_name, "model_cfg.json")
            os.remove(temp_name)

        callback_func = None
        if should_prune and args.prune_aware:
            callback_func = do_pruning

        distill_model(task, model, device, args, callback_func, sacred_experiment)

    if should_prune:
        # Magnitude pruning after distillation (static).
        model_name = args.load_trained_model
        if model is None:
            model = load_student(task, student_type, use_gpu=use_gpu, model_name=model_name)
        model.to(device)
        model = prune_model(model, device, args)

    if should_quantize:
        use_gpu = False
        device = torch.device('cpu')
        model_name = args.load_trained_model
        if model is None:
            model = load_student(task, student_type, use_gpu=use_gpu, model_name=model_name)
        model.cfg['use-gpu'] = False
        model.to(device)
        quantize_model(model, device, args)

if __name__ == "__main__":
    ARGS, REMAIN = argparsers.args_compress()
    if len(REMAIN) > 0:
        raise ArgumentError(None, f"Couldn't parse the following arguments: {REMAIN}")
    main(ARGS)
