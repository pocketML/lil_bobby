from analysis import parameters, pretty_print, plotting
from compression.distillation import models as distill_models
from compression import quantize
from common import argparsers, model_utils
from fairseq.models.roberta import RobertaModel
from custom import glue_bilstm

def main(args, sacred_experiment=None):
    is_roberta_model = model_utils.is_finetuned_model(args.arch)
    if is_roberta_model:
        if args.non_finetuned:
            model = model_utils.load_roberta_model(args.arch, use_cpu=True)
        else:
            model_path = model_utils.get_model_path(args.task, "finetuned")
            model = model_utils.load_teacher(args.task, f"{model_path}/{args.model_name}", use_cpu=True)
        
    elif args.arch in distill_models.STUDENT_MODELS.keys():
        model = distill_models.load_student(args.task, args.arch, False, model_name=args.model_name)
    elif args.arch == 'glue':
        model = glue_bilstm.Model()

    model.eval()
    if args.model_disk_size:
        pretty_print.print_model_disk_size(model)
    if args.model_size:
        pretty_print.print_model_size(model)
        total_params, total_bits = parameters.get_model_size(model)
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("model_params", total_params)
            sacred_experiment.log_scalar("model_size", total_bits/8000000)
    if args.layer_weight_hist:
        pass
    if args.named_params:
        pretty_print.print_named_params(model, args.arch)
    if args.weight_thresholds:
        pretty_print.print_threshold_stats(model, args.arch)
    if args.weight_hist and is_roberta_model:
        plotting.weight_histogram_for_all_transformers(model, args.arch)
    if args.pie_chart:
        plotting.weight_pie_chart(model, args.arch)

if __name__ == "__main__":
    ARGS = argparsers.args_analyze()

    main(ARGS)
