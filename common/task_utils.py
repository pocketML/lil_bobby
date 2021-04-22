GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data"

TASK_LABEL_DICT = {
    'sst-2': {
        '1': 0,
        '0': 1,
    },
    'mnli': {
        'contradiction': 0,
        'neutral': 1,
        'entailment': 2,
    },
    'qqp': {
        '0': 0,
        '1': 1,
    }
}

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
            'use-sentence-pairs' : True,
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
            'use-sentence-pairs' : True,
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
            'use-sentence-pairs' : True,
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
            'use-sentence-pairs' : True,
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
            'use-sentence-pairs' : False,
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

SEED_DICT = {
    "hadfield": 8291959,
    "knuth": 1101938,
    "feynman": 5111918,
    "miyazaki": 1051941,
    "doom": 7131971,
    "bennington": 3201976,
    "lorca": 6051898,
    "armstrong": 8051930,
    "cardi": 10111992,
    "simone": 2211933,
    "lovelace": 12101815,
    "shannon": 4301916
}

def get_processed_path(task):
    return f'{TASK_INFO[task]["path"]}/processed/{task}-bin/'

def is_sentence_pair(task):
    return TASK_INFO[task]['settings']['use-sentence-pairs']