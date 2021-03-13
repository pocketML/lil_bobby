GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data"

TASK_INFO = {
    'mnli' : {
        'path': 'data/glue/MNLI',
        'download_url': [f'{GLUE_URL}/MNLI.zip'],
        'settings': {
            'num-classes' : 3,
            'lr' : 1e-5,
            'batch-size': 32,
            'total-num-update': 123873,
            'warmup-updates': 7432,
        }
    },
    'qnli' : {
        'path': 'data/glue/QNLI',
        'download_url': [f'{GLUE_URL}/QNLI.zip'],
        'settings': {
            'num-classes' : 2,
            'lr' : 1e-5,
            'batch-size': 32,
            'total-num-update': 33112,
            'warmup-updates': 1986,
        }
    },
    'qqp' : {
        'path': 'data/glue/QQP',
        'download_url': [f'{GLUE_URL}/QQP.zip'],
        'settings': {
            'num-classes' : 2,
            'lr' : 1e-5,
            'batch-size': 32,
            'total-num-update': 113272,
            'warmup-updates': 28318,
        }
    },
    'rte' : {
        'path': 'data/glue/RTE',
        'download_url': [f'{GLUE_URL}/RTE.zip'],
        'settings': {
            'num-classes' : 2,
            'lr' : 2e-5,
            'batch-size': 16,
            'total-num-update': 2036,
            'warmup-updates': 122,
        }
    },
    'sst-2' : {
        'path': 'data/glue/SST-2',
        'download_url': [f'{GLUE_URL}/SST-2.zip'],
        'settings': {
            'num-classes' : 2,
            'lr' : 1e-5,
            'batch-size': 32,
            'total-num-update': 20935,
            'warmup-updates': 1256,
        }
    },
    'mrpc' : {
        'path': 'data/glue/MRPC',
        'download_url': [
            'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt',
            'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt',
            "https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv"
        ],
        'settings': {
            'num-classes' : 2,
            'lr' : 1e-5,
            'batch-size': 16,
            'total-num-update': 2296,
            'warmup-updates': 137,
        }
    },
    'cola' : {
        'path': 'data/glue/CoLA',
        'download_url': [f'{GLUE_URL}/CoLA.zip'],
        'settings': {
            'num-classes' : 2,
            'lr' : 1e-5,
            'batch-size': 16,
            'total-num-update': 5336,
            'warmup-updates': 320,
        }
    },
    'sts-b' : {
        'path': 'data/glue/STS-B',
        'download_url': [f'{GLUE_URL}/STS-B.zip'],
        'settings': {
            'num-classes' : 1,
            'lr' : 2e-5,
            'batch-size': 16,
            'total-num-update': 3598,
            'warmup-updates': 214,
        }
    },
    'ax' : {
        'path': 'data/glue/AX',
        'download_url': [f'{GLUE_URL}/AX.tsv'],
        'settings': {
            'num-classes' : 3,
            'lr' : 1e-5,
            'batch-size': 32,
            'total-num-update': 123873,
            'warmup-updates': 7432,
        }
    }
}

def get_processed_path(task):
    return f'{TASK_INFO[task]["path"]}/processed/{task}-bin/'

def get_finetune_string(
    task_path, model_path, override_args, sacred_experiment=None):
    task = override_args.task
    arch = override_args.arch
    use_fp16 = override_args.fp16
    use_cpu = override_args.cpu
    settings = TASK_INFO[task]['settings']
    
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
        '--model-parallel-size', f'{gpus}'
    ]
    if task == 'sts-b':
        arguments.extend(['--regression-target', '--best-checkpoint-metric', 'loss'])
    else:
        arguments.extend(['--best-checkpoint-metric', 'accuracy', '--maximize-best-checkpoint-metric'])
    if use_fp16 and not use_cpu:
        arguments.extend([
            '--fp16',
            '--fp16-init-scale', '4',
            '--fp16-scale-window', '128',])
    if use_cpu:
        arguments.extend(['--cpu'])
    if sacred_experiment is not None:
        experiment_name = sacred_experiment.info["name"]
        checkpoint_dir = f"checkpoints/{experiment_name}"
        arguments.extend(['--save-dir', checkpoint_dir])
    return arguments
