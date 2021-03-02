import argparse
from fairseq import options
from common.task_utils import TASK_INFO
from common.model_utils import MODEL_INFO
from download import get_dataset_path

pipeline_tasks = [
    'finetune',
    'prune-magnitude',
    'quantize'
]

finetune_tasks = list(TASK_INFO.keys())

# define arguments for model compression
def args_compress():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=pipeline_tasks, nargs="+")
    ap.add_argument("--finetune-before", "-ftb", choices=finetune_tasks)
    ap.add_argument("--finetune-during", "-ftd", choices=finetune_tasks)
    ap.add_argument("--finetune-after", "-fta", choices=finetune_tasks)
    ap.add_argument('--config')
    args = ap.parse_args()
    return args

# download benchmark tasks, roberta models etc
def args_download():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", "-t", choices=list(TASK_INFO.keys()) + ["glue"])
    group.add_argument("--model", "-m", choices=MODEL_INFO.keys())
    args = parser.parse_args()
    return args

def args_evaluate():
    pass

def args_experiment():
    pass

def args_finetune():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--task", "-t", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--model", "-m", choices=MODEL_INFO.keys(), required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    #ap.add_argument("--config", "-config", required=True)
    
    args = ap.parse_args()
    return args

def args_analyze():
    models = [k for k in MODEL_INFO.keys()]
    [models.append(k) for k in MODEL_INFO.keys()]
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=models, required=True)
    ap.add_argument('--model-path', type=str)
    ap.add_argument('--model-size', action='store_true')
    ap.add_argument('--weight-hist', action='store_true')
    ap.add_argument('--layer-weight-hist', type=str)
    

    args = ap.parse_args()
    return args

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
