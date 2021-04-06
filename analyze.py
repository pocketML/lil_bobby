from fairseq.models.roberta import RobertaModel
from analysis import parameters
from common import argparsers, task_utils, model_utils
from compression.distillation import models as distill_models

def main(args, sacred_experiment=None):
    data_path = task_utils.get_processed_path(args.task)
    finetuned_model = args.arch in model_utils.MODEL_INFO.keys()

    if finetuned_model:
        model = RobertaModel.from_pretrained(
            'checkpoints',
            checkpoint_file=args.model_name,
            data_name_or_path=data_path
        )
    else: # is in compressions.distillation.models.STUDENT_MODELS.keys()
        model = distill_models.load_student(args.task, args.arch, False, model_name=args.model_name)

    model.eval()

    if args.model_size:
        parameters.print_model_size(model)
    if args.weight_hist and finetuned_model:
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
