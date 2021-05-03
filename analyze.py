from analysis import parameters
from compression.distillation import models as distill_models
from compression import quantize
from common import argparsers, model_utils

def main(args, sacred_experiment=None):
    is_finetuned_model = model_utils.is_finetuned_model(args.arch)
    if is_finetuned_model:
        model_path = model_utils.get_model_path(args.task, "finetuned")
        model = model_utils.load_teacher(args.task, f"{model_path}/{args.model_name}", use_cpu=True)
    else: # is in compressions.distillation.models.STUDENT_MODELS.keys()
        model = distill_models.load_student(args.task, args.arch, False, model_name=args.model_name)

    model.eval()
    if args.model_disk_size:
        model_static_quant = quantize.quantize_model(model)
        parameters.print_model_disk_size(model)
        parameters.print_model_disk_size(model_static_quant)
        print(model)
        print(model_static_quant)
    if args.model_size:
        parameters.print_model_size(model)
        total_params, total_bits = parameters.get_model_size(model)
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("model_params", total_params)
            sacred_experiment.log_scalar("model_size", total_bits/8000000)
    if args.weight_hist and is_finetuned_model:
        parameters.weight_histogram_for_all_transformers(model, args.arch)
    if args.layer_weight_hist:
        pass
    if args.named_params:
        parameters.print_named_params(model, args.arch)
    if args.weight_thresholds:
        parameters.print_threshold_stats(model, args.arch)

if __name__ == "__main__":
    ARGS = argparsers.args_analyze()

    main(ARGS)
