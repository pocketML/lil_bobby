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
            f'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt',
            f'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
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

def get_finetune_string(task, task_path, model_path, batch_size, use_fp16, arch='roberta_base'):
    settings = TASK_INFO[task]['settings']
    data_path = f'{task_path}/processed/{task}-bin/'
    update_freq = int(settings['batch-size'] / batch_size)
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
        '--adam-betas', "(0.9,0.98)",
        '--adam-eps', '1e-06',
        '--clip-norm', '0.0',
        '--lr-scheduler', 'polynomial_decay',
        '--lr', f'{settings["lr"]}',
        '--total-num-update', f'{settings["total-num-update"]}',
        '--warmup-updates', f'{settings["warmup-updates"]}',
        '--threshold-loss-scale', '1',
        '--max-epoch', '10',
        '--find-unused-parameters',
        '--update-freq', f'{update_freq}'
    ]
    if task == 'sts-b':
        arguments.extend(['--regression-target', '--best-checkpoint-metric', 'loss'])
    else:
        arguments.extend(['--best-checkpoint-metric', 'accuracy', '--maximize-best-checkpoint-metric'])
    if use_fp16:
        arguments.extend([
            '--fp16',
            '--fp16-init-scale', '4',
            '--fp16-scale-window', '128',])
    return arguments
