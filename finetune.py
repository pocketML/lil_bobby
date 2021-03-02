from common import argparsers, task_utils
from download import get_dataset_path, get_model_path
from roberta_custom.train import cli_main

if __name__ == "__main__":
    ARGS = argparsers.args_finetune()

    task = ARGS.task
    model = ARGS.model
    task_path = get_dataset_path(task)
    model_path = get_model_path(model)
    batch_size = ARGS.batch_size

    finetune_args = task_utils.get_finetune_string(task, task_path, model_path, batch_size)
    print(finetune_args)

    cli_main(input_args=finetune_args)
