from argparse import ArgumentError

from analysis import parameters, pretty_print, plotting
from compression.distillation import models as distill_models
from common import argparsers, model_utils

def load_model(args, is_roberta_model, kw_model=None):
    if kw_model is not None:
        return kw_model
    elif is_roberta_model:
        if args.non_finetuned:
            return model_utils.load_roberta_model(args.arch, use_cpu=True)
        else:
            model_path = model_utils.get_model_path(args.task, "finetuned")
            return model_utils.load_teacher(args.task, f"{model_path}/{args.model_name}", use_cpu=True)
    elif args.arch == 'glue':
        # 2_200_000 vocab_size for original GloVe
        # 312_000 for ELMO
        return model_utils.GlueBaseline(vocab_size=312000)
    elif args.arch in distill_models.STUDENT_MODELS.keys():
        return distill_models.load_student(args.task, args.arch, False, model_name=args.model_name, args=args)

def main(args, **kwargs):
    sacred_experiment = kwargs.get("sacred_experiment")
    is_roberta_model = model_utils.is_finetuned_model(args.arch)
    model = load_model(args, is_roberta_model, kwargs.get("model"))
    model.eval()
    if args.model_disk_size:
        pretty_print.print_model_disk_size(model, sacred_experiment)
        if sacred_experiment is not None:
            disk_size = parameters.get_model_disk_size(model, sacred_experiment)
            sacred_experiment.log_scalar("model_disk_size", disk_size)
    if args.model_size:
        pretty_print.print_model_size(model)
        total_params, total_bits = parameters.get_model_size(model)
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("model_params", total_params)
            sacred_experiment.log_scalar("model_size", total_bits/8000000)
    if args.theoretical_size:
        pretty_print.print_theoretical_size(model, sacred_experiment)
        total_params, total_bits = parameters.get_theoretical_size(model, sacred_experiment)
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("nonzero_params", total_params)
            sacred_experiment.log_scalar("theoretical_size", total_bits)
    if args.named_params:
        pretty_print.print_named_params(model, args.arch)
    if args.weight_thresholds:
        pretty_print.print_threshold_stats(model, args.arch)
    if args.weight_hist and is_roberta_model:
        plotting.weight_histogram_for_all_transformers(model, args.arch)
    if args.pie_chart:
        plotting.weight_pie_chart(model, args.arch, args.save_pdf)
    return model

if __name__ == "__main__":
    ARGS, REMAIN = argparsers.args_analyze()
    if len(REMAIN) > 0:
        raise ArgumentError(None, f"Couldn't parse the following arguments: {REMAIN}")
    main(ARGS)
