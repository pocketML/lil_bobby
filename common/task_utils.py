TASK_INFO = {
    'mnli' : 
        'path': 'data/glue/MNLI',
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
        'settings': {
            'num-classes' : 1,
            'lr' : 2e-5,
            'batch-size': 16,
            'total-num-update': 3598,
            'warmup-updates': 214,
        }
    },
}

def get_finetune_string(task, model_path, arch='roberta_base'):
    settings = TASK_INFO[task]['settings']
    arguments = [
        '--restore-file', f'{model_path}',
        '--max-positions', '512',
        '--batch-size', f'{settings["batch-size"]}',
        '--max-tokens', '4400',
        '--task', 'sentence_prediction',
        '--reset-optimizer', 
        '--reset-dataloaders', 
        '--reset-meters',
        '--required-batch-size-multiple', '1',
        '--init-token', '0',
        '--separator-token', '2',
        '--arch', f'{arch}',
        '--criterion', 'sentence_prediction',
        '--num-classes', f'{settings["num_classes"]}',
        '--dropout', '0.1',
        '--attention-dropout', '0.1',
        '--weight-decay', '0.1',
        '--optimizer', 'adam',
        '--adam-betas', '"(0.9, 0.98)"',
        '--adam-eps', '1e-06',
        '--clip-norm', '0.0',
        '--lr-scheduler', 'polynomial_decay',
        '--lr', f'{settings["lr"]}',
        '--total-num-update' f'{settings["total-num-updates"]}',
        '--warmup-updates', f'{settings["warmup-updates"]}',
        '--fp16',
        '--fp16-init-scale', '4',
        '--threshold-loss-scale', '1',
        '--fp16-scale-window', '128',
        '--max-epochs', '10',
        '--find-unused-parameters',
    ]
    if task == 'sts-b':
        arguments.extend(['--regression-target', '--best-checkpoint-metric', 'loss'])
    else:
        arguments.extend(['--best-checkpoint-metric', 'accuracy', '--maximize-best-checkpoint-metric'])
    return arguments
