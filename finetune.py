from common import argparsers, task_utils, model_utils
from download import get_dataset_path, get_roberta_path
from roberta_custom import train


def get_finetune_string(
    task_path, model_path, override_args, sacred_experiment=None):
    task = override_args.task
    arch = override_args.arch
    use_fp16 = override_args.fp16
    use_cpu = override_args.cpu
    settings = task_utils.TASK_INFO[task]['settings']
    seed = override_args.seed
    
    # setting batch_size to task default if nothing specified
    batch_size = override_args.batch_size
    batch_size = batch_size if batch_size is not None else settings['batch-size']

    data_path = f'{task_path}/processed/{task}-bin/'
    update_freq = int(settings['batch-size'] / batch_size)
    max_epochs = override_args.max_epochs
    gpus = override_args.model_parallel_size

    arguments = [
        f'{data_path}', # FILE
        '--restore-file', f'{model_path}',
        '--max-positions', '512',
        '--batch-size', f'{batch_size}',
        '--max-tokens', '4400',
        '--task', 'sentence_prediction',
        '--reset-optimizer',
        '--reset-dataloader',
        '--reset-meters',
        '--required-batch-size-multiple', '1',
        '--init-token', '0',
        '--separator-token', '2',
        '--arch', f'{arch}',
        '--criterion', 'sentence_prediction',
        '--num-classes', f'{settings["num-classes"]}',
        '--dropout', '0.1',
        '--attention-dropout', '0.1',
        '--weight-decay', '0.1',
        '--optimizer', 'adam',
        '--adam-betas', "(0.9, 0.98)",
        '--adam-eps', '1e-06',
        '--clip-norm', '0.0',
        '--lr-scheduler', 'polynomial_decay',
        '--lr', f'{settings["lr"]}',
        '--total-num-update', f'{settings["total-num-update"]}',
        '--warmup-updates', f'{settings["warmup-updates"]}',
        '--threshold-loss-scale', '1',
        '--max-epoch', f"{max_epochs}",
        '--find-unused-parameters',
        '--update-freq', f'{update_freq}',
        '--no-epoch-checkpoints',
        '--model-parallel-size', f'{gpus}',
        '--seed', f'{seed}',
        '--no-save-optimizer-state'
    ]
    if task == 'sts-b':
        arguments.extend(['--regression-target', '--best-checkpoint-metric', 'loss'])
    else:
        arguments.extend([
            '--best-checkpoint-metric', 'accuracy',
            '--maximize-best-checkpoint-metric'
        ])

    if use_fp16 and not use_cpu:
        arguments.extend([
            '--fp16',
            '--fp16-init-scale', '4',
            '--fp16-scale-window', '128',])

    if use_cpu:
        arguments.extend(['--cpu'])

    if sacred_experiment is not None:
        experiment_name = sacred_experiment.info["name"]
        base_dir = model_utils.get_model_path(task, "finetuned")
        checkpoint_dir = f"{base_dir}/{experiment_name}"
        arguments.extend(['--save-dir', checkpoint_dir])

    return arguments


def main(args, sacred_experiment=None):
    task = args.task
    task_path = get_dataset_path(task)
    if args.arch == 'roberta_base':
        model_path = get_roberta_path('base') + '/model.pt'
    else:
        model_path = get_roberta_path('large') + '/model.pt'

    finetune_args = get_finetune_string(
        task_path, model_path, args,
        sacred_experiment=sacred_experiment
    )
    print(finetune_args)

    train.cli_main(input_args=finetune_args, sacred_experiment=sacred_experiment)

if __name__ == "__main__":
    ARGS = argparsers.args_finetune()

    main(ARGS)
