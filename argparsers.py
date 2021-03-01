import argparse
from fairseq import options
from download import get_dataset_path, TASK_DATASET_PATHS, MODEL_PATHS

pipeline_tasks = [
    'finetune',
    'prune-magnitude',
    'quantize'
]

finetune_tasks = list(TASK_DATASET_PATHS.keys())

# define arguments for model compression
def get_argparser_compress():
    a = argparse.ArgumentParser()
    a.add_argument("task", choices=pipeline_tasks, nargs="+")
    a.add_argument("--finetune-before", "-ftb", choices=finetune_tasks)
    a.add_argument("--finetune-during", "-ftd", choices=finetune_tasks)
    a.add_argument("--finetune-after", "-fta", choices=finetune_tasks)
    a.add_argument('--config')
    return a

# download benchmark tasks, roberta models etc
def get_argparser_download():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", "-t", choices=TASK_DATASET_PATHS.keys())
    group.add_argument("--model", "-m", choices=MODEL_PATHS.keys())
    return parser

def get_argparser_evaluate():
    pass

def get_argparser_experiment():
    pass

def get_argparser_finetune():
    a = argparse.ArgumentParser()
    a.add_argument("--config", "-config", required=True)
    return a

def get_argparser_inspect():
    pass

def parse_roberta_args(parser):
    args = parser.parse_args()

    input_args = load_config_file(args.config)

    try:
        dataset = get_dataset_path(args.finetune_before)
    except KeyError:
        print(f"Error: Task '{args.finetune_before}' is not valid.")
        exit(0)

    input_args.append(dataset)
    input_args.extend(["--task", args.finetune_before])

    roberta_parser = options.get_training_parser()
    return options.parse_args_and_arch(roberta_parser, input_args=input_args)

def load_config_file(filename):
    with open(filename, encoding="utf-8") as fp:
        args = []
        for line in fp:
            stripped = line.strip()
            args.extend(stripped.split(" "))
        return args
