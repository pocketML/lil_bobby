from common import argparsers, task_utils
from download import get_dataset_path, get_model_path
from roberta_custom.train import cli_main

if __name__ == "__main__":
    ARGS = argparsers.args_finetune()

    TASK = ARGS.task
    MODEL = ARGS.model
    TASK_PATH = get_dataset_path(task)
    MODEL_PATH = get_model_path(model)
    BATCH_SIZE = ARGS.batch_size
    USE_FP16 = not ARGS.cpu

    FINETUNE_ARGS = task_utils.get_finetune_string(TASK, TASK_PATH, MODEL_PATH, BATCH_SIZE, USE_FP16)
    print(finetune_args)

    cli_main(input_args=finetune_args)
