from argparse import ArgumentError

from analysis import parameters, pretty_print, plotting
from compression.distillation import models as distill_models
from common import argparsers, model_utils

def load_model(args, is_roberta_model, kw_model=None):
    """
    Load a baseline/finetuned/distilled model.

    Parameters
    ----------
    args : Namespace
        Arguments for what type of model to load and where it is saved.
    is_roberta_model : bool
        Whether the model to load is a RoBERTa model (either a masked LM or finetuned).
    kw_model : Model
        This may be an existing model or None. If it is not None, return it.

    Returns
        A loaded model, either a masked LM, finetuned, or distilled model.
    ----------
    """
    if kw_model is not None:
        return kw_model
    elif is_roberta_model: # Load RoBERTa model.
        if args.non_finetuned: # Load roberta_base or roberta_large masked LM.
            return model_utils.load_roberta_model(args.arch, use_cpu=True)
        else: # Load model finetuned for specific task.
            model_path = model_utils.get_model_path(args.task, "finetuned")
            return model_utils.load_teacher(args.task, f"{model_path}/{args.model_name}", use_cpu=True)
    elif args.arch in ('glue_glove', 'glue_elmo'): # Load GLUE baseline model.
        # 2_200_000 vocab_size for original GloVe
        # 312_000 for ELMO
        vocab_size = 2200000 if args.arch == "glue_glove" else 312000
        return model_utils.GlueBaseline(args.task, vocab_size=vocab_size)
    elif args.arch in distill_models.STUDENT_MODELS.keys(): # Load distilled model.
        return distill_models.load_student(args.task, args.arch, False, model_name=args.model_name, args=args)

def main(args, **kwargs):
    # Get Sacred experiment instance (if it exists).
    sacred_experiment = kwargs.get("sacred_experiment")
    # Determine whether the model we should load is a RoBERTa or distilled model.
    is_roberta_model = model_utils.is_finetuned_model(args.arch)
    model = load_model(args, is_roberta_model, kwargs.get("model"))
    model.eval()
    if args.model_disk_size: # Print disk size of model.
        pretty_print.print_model_disk_size(model, sacred_experiment)
        if sacred_experiment is not None:
            disk_size = parameters.get_model_disk_size(model, sacred_experiment)
            sacred_experiment.log_scalar("model_disk_size", disk_size)
    if args.model_size: # Print parameter size of model.
        pretty_print.print_model_size(model)
        total_params, total_bits = parameters.get_model_size(model)
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("model_params", total_params)
            sacred_experiment.log_scalar("model_size", total_bits/8000000)
    if args.theoretical_size: # Print zipped disk size of model.
        pretty_print.print_theoretical_size(model, sacred_experiment)
        total_params, total_bits = parameters.get_theoretical_size(model, sacred_experiment)
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("nonzero_params", total_params)
            sacred_experiment.log_scalar("theoretical_size", total_bits)
    if args.named_params: # Print an overview of parameters/layers in the model.
        pretty_print.print_named_params(model, args.arch)
    if args.weight_thresholds: # Print histogram of weight magnitudes under certain thresholds.
        pretty_print.print_threshold_stats(model, args.arch)
    if args.weight_hist and is_roberta_model:
        # Print histograms of weights in transformer layers for a RoBERTa model.
        plotting.weight_histogram_for_all_transformers(model, args.arch)
    if args.pie_chart: # Print pie chart of parameter counts in different areas of a model.
        plotting.weight_pie_chart(model, args.arch, args.save_pdf)
    return model

if __name__ == "__main__":
    ARGS, REMAIN = argparsers.args_analyze()
    if len(REMAIN) > 0:
        raise ArgumentError(None, f"Couldn't parse the following arguments: {REMAIN}")
    main(ARGS)
