from analysis import parameters
from compression.distillation import models as distill_models
from compression.quantization import post_training
from common import argparsers, model_utils

def main(args, sacred_experiment=None):
    is_finetuned_model = model_utils.is_finetuned_model(args.arch)
    if is_finetuned_model:
        model = model_utils.load_teacher(args.task, 'checkpoints', use_cpu=True, model_name=args.model_name)
    else: # is in compressions.distillation.models.STUDENT_MODELS.keys()
        model = distill_models.load_student(args.task, args.arch, False, model_name=args.model_name)

    model.eval()
    if args.model_disk_size:
        model_static_quant = post_training.quantize_model(model)
        parameters.print_model_disk_size(model)
        parameters.print_model_disk_size(model_static_quant)
        print(model)
        print(model_static_quant)
    if args.model_size:
        parameters.print_model_size(model)
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
