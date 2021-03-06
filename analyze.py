from fairseq.models.roberta import RobertaModel
from analysis import parameters
from common import argparsers, task_utils

# weight_histogram_for_all_transformers(roberta)
#parameters.print_threshold_stats(roberta)
#parameters.print_threshold_stats(roberta)
#parameters.print_model_size(roberta)

if __name__ == "__main__":
    ARGS = argparsers.args_analyze()
    DATA_PATH = task_utils.get_processed_path(ARGS.task)
    MODEL = RobertaModel.from_pretrained(
        'checkpoints',
        checkpoint_file=ARGS.model_name,
        data_name_or_path=DATA_PATH
    )
    MODEL.eval()

    if ARGS.model_size:
        parameters.print_model_size(MODEL)
    if ARGS.weight_hist:
        parameters.weight_histogram_for_all_transformers(MODEL)
    if ARGS.layer_weight_hist:
        pass
    if ARGS.named_params:
        parameters.print_named_params(MODEL)
    if ARGS.weight_thresholds:
        parameters.print_threshold_stats(MODEL)