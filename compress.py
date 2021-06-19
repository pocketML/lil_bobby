import torch
import torch.nn as nn

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
    sacred_experiment = kwargs.get("sacred_experiment")
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    use_gpu = not args.cpu
    task = args.task
    student_type = args.student_arch

    seed_utils.set_global_seed(args.seed, set_cuda_deterministic=True)

    should_prune = "prune" in args.compression_actions
    should_quantize = "quantize" in args.compression_actions

    if "distill" in args.compression_actions:
        model = models.load_student(task, student_type, use_gpu=use_gpu, args=args)
        if sacred_experiment is not None:
            temp_name = f"temp_{sacred_experiment.info['name']}.json"
            args.load_trained_model = sacred_experiment.info['name']
            with open(temp_name, "w", encoding="utf-8") as fp:
                json.dump(model.cfg, fp, indent=4)
            sacred_experiment.add_artifact(temp_name, "model_cfg.json")
            os.remove(temp_name)
        else:
            args.load_trained_model = args.student_arch

        callback_func = prune.do_pruning if should_prune and args.prune_aware else None
        distill.distill_model(task, model, device, args, callback_func, sacred_experiment)

    if should_prune:
        if not args.prune_aware:
            # Magnitude pruning after distillation (static).
            model = models.load_student(task, student_type, use_gpu=use_gpu, model_name=args.load_trained_model)
            model.to(device)
            model = prune.prune_model(model, device, args, sacred_experiment)

    if should_quantize:
        use_gpu = False
        device = torch.device('cpu')
        if not should_prune:
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
