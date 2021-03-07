from common import argparsers, task_utils
from download import get_dataset_path, get_model_path
from roberta_custom.train import cli_main

def main(args, sacred_experiment=None):
    task = args.task
    model = args.model
    task_path = get_dataset_path(task)
    model_path = get_model_path(model)
    batch_size = args.batch_size
    use_fp16 = not args.cpu

    finetune_args = task_utils.get_finetune_string(task, task_path, model_path, batch_size, use_fp16)
    print(finetune_args)

    cli_main(input_args=finetune_args, sacred_experiment=sacred_experiment)

if __name__ == "__main__":
    ARGS = argparsers.args_finetune()

    main(ARGS)
