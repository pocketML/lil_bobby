from fairseq.models.roberta import RobertaModel
from analysis import parameters
from common import argparsers, task_utils

def main(args, sacred_experiment=None):
    data_path = task_utils.get_processed_path(args.task)
    model = RobertaModel.from_pretrained(
        'checkpoints',
        checkpoint_file=args.model_name,
        data_name_or_path=data_path
    )
    model.eval()

    if args.model_size:
        parameters.print_model_size(model)
    if args.weight_hist:
        parameters.weight_histogram_for_all_transformers(model)
    if args.layer_weight_hist:
        pass
    if args.named_params:
        parameters.print_named_params(model)
    if args.weight_thresholds:
        parameters.print_threshold_stats(model)

if __name__ == "__main__":
    ARGS = argparsers.args_analyze()

    main(ARGS)    
