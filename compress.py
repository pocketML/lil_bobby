import torch

from argparse import ArgumentError
import json
import os
import warnings

from common import argparsers, seed_utils
from compression.distillation import models
from compression import quantize, distill, prune
from compression import prune

warnings.simplefilter("ignore", UserWarning)

def main(args, **kwargs):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life")
    # Get Sacred experiment instance (if it exists) and other values from 'args'.
    sacred_experiment = kwargs.get("sacred_experiment")
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    task = args.task
    student_type = args.student_arch

    # Set seed for Python, Numpy, PyTorch random number generators.
    seed_utils.set_global_seed(args.seed, set_cuda_deterministic=True)

    # Determine whether we should do pruning or quantization.
    should_prune = "prune" in args.compression_actions
    should_quantize = "quantize" in args.compression_actions

    if "distill" in args.compression_actions: # Do distillation.
        model = models.load_student(task, student_type, use_gpu=use_gpu, args=args)
        if sacred_experiment is not None:
            # Create temporary json file with model configs.
            temp_name = f"temp_{sacred_experiment.info['name']}.json"
            # Set the name of the trained model to the name of the active Sacred experiment.
            args.load_trained_model = sacred_experiment.info['name']
            # Save model config to temporary json file.
            with open(temp_name, "w", encoding="utf-8") as fp:
                json.dump(model.cfg, fp, indent=4)
            # Save temporary json config file to Sacred experiment folder.
            sacred_experiment.add_artifact(temp_name, "model_cfg.json")
            os.remove(temp_name)
        else:
            args.load_trained_model = args.student_arch

        # If we should do training-aware pruning, define a callback 
        # function for pruning, called after each epoch during distillation.
        callback_func = prune.do_pruning if should_prune and args.prune_aware else None
        model = distill.distill_model(task, model, device, args, callback_func, sacred_experiment)

    if should_prune and not args.prune_aware:
        # Post-training pruning (static). Load a model and prune it.
        model = models.load_student(task, student_type, use_gpu=use_gpu, model_name=args.load_trained_model)
        model.to(device)
        model = prune.prune_model(model, device, args, sacred_experiment)

    if should_quantize:
        # Post-training quantization. Quantization only works on CPU.
        use_gpu = False
        device = torch.device('cpu')
        if not should_prune: # If we already pruned, don't load a new model in.
            model = models.load_student(task, student_type, use_gpu=use_gpu, model_name=args.load_trained_model)
        model.cfg['use-gpu'] = False
        model.to(device)
        model = quantize.quantize_model(model, device, args, sacred_experiment)

    # Check if we did post-training compression and should save a compressed model.
    pt_pruning = should_prune and not args.prune_aware
    if (should_quantize or pt_pruning) and sacred_experiment is not None:
        model_name = sacred_experiment.info["name"]
        model.save(model_name)

    return model

if __name__ == "__main__":
    ARGS, REMAIN = argparsers.args_compress()
    if len(REMAIN) > 0:
        raise ArgumentError(None, f"Couldn't parse the following arguments: {REMAIN}")
    main(ARGS)
