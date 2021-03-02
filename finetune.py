from common import argparsers, task_utils
from common.model_utils import MODEL_INFO
from roberta_custom.train import cli_main
from pathlib import Path

if __name__ == "__main__":
    ARGS = argparsers.args_finetune()

    task = ARGS.task
    model_path = MODEL_INFO[ARGS.model]['path']
    finetune_args = task_utils.get_finetune_string(task, model_path)

    cli_main(input_args=finetune_args)