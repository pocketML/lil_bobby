from argparse import ArgumentError
import json
import os
import torch
import torch.nn as nn
from common import argparsers, data_utils
from compression.distill import train_loop
from compression.distillation.models import DistLossFunction, load_student
import evaluate
from compression import prune
from compression import quantize as ptq
from analysis import parameters
import warnings
import random

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

    print(model)

    evaluate.evaluate_distilled_model(model, dl, device, args, None)
    parameters.print_model_disk_size(model)
    print()

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

def distill_model(task, model, device, args, callback, sacred_experiment):
    epochs = args.epochs
    temperature = args.temperature
    model.to(device)

    distillation_data = data_utils.load_all_distillation_data(
        task, only_original_data=args.original_data, chunk_size=args.chunk_size
    )
    print(f"*** Loaded {len(distillation_data[0])} training data samples ***")
    
    val_data = data_utils.load_val_data(task)
    print(f"*** Loaded {len(val_data[0])} validation data samples ***")

    criterion = DistLossFunction(
        args.alpha, 
        nn.MSELoss(), 
        nn.CrossEntropyLoss(), 
        device,
        temperature=temperature,
    )
    
    dataloaders = data_utils.get_dataloader_dict(model, distillation_data, val_data)
    print(f'*** Dataloaders created ***')

    optim = model.get_optimizer()
    train_loop(
        model, criterion, optim, dataloaders, device,
        args, epochs, compress_callback=callback,
        sacred_experiment=sacred_experiment
    )

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life")

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    task = args.task
    student_type = args.student_arch
    seed = args.seed

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    should_prune = "prune" in args.compression_actions
    should_quantize = "quantize" in args.compression_actions

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

    if should_quantize:
        use_gpu = False
        device = torch.device('cpu')
        model_name = args.load_trained_model
        model = load_student(task, student_type, use_gpu=use_gpu, model_name=model_name)
        model.to(device)
        quantize_model(model, device, args)

    if should_prune:
        # Magnitude pruning after distillation (static).
        model_name = args.load_trained_model
        model = load_student(task, student_type, use_gpu=use_gpu, model_name=model_name)
        model.to(device)
        prune_model(model, device, args)

if __name__ == "__main__":
    ARGS, remain = argparsers.args_compress()
    if len(remain) > 0:
        raise ArgumentError(None, f"Couldn't parse the following arguments: {remain}")
    main(ARGS)
