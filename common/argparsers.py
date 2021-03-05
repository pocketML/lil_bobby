import argparse
from re import sub
from fairseq import options
from common.task_utils import TASK_INFO
from common.model_utils import MODEL_INFO
from download import get_dataset_path

compression_techniques = [
    'prune-magnitude',
    'quantize'
]

finetune_tasks = list(TASK_INFO.keys())

# define arguments for model compression
def args_compress(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--techniques", choices=compression_techniques, nargs="+", required=True)
    ap.add_argument("--pruning-threshold", type=float)
    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

# download benchmark tasks, roberta models etc
def args_download():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", "-t", choices=list(TASK_INFO.keys()) + ["glue"])
    group.add_argument("--model", "-m", choices=MODEL_INFO.keys())
    return ap

def args_evaluate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument('--cpu', action='store_true')

    return ap

def args_finetune(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", "-t", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--model", "-m", choices=MODEL_INFO.keys(), required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--use-fp16", action='store_true')
    #ap.add_argument("--config", "-config", required=True)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

def args_analyze():
    models = list(MODEL_INFO.keys())
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=models, required=True)
    ap.add_argument('--model-path', type=str)
    ap.add_argument('--model-size', action='store_true')
    ap.add_argument('--weight-hist', action='store_true')
    ap.add_argument('--layer-weight-hist', type=str)
    ap.add_argument('--named-params', action='store_true')
    return ap

def args_experiment():
    ap = argparse.ArgumentParser()

    task_choices = ("finetune", "compress", "evaluate", "analyze")
    ap.add_argument("tasks", nargs="+", choices=task_choices)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--output-path", type=str)

    ex_args = argparse.Namespace()

    ex_args, args_remain = ap.parse_known_args(namespace=ex_args)
    if "finetune" in ex_args.tasks:
        ex_args, args_remain = args_finetune(args_remain, namespace=ex_args, parse_known=True)
    if "compress" in ex_args.tasks:
        ex_args, args_remain = args_compress(args_remain, namespace=ex_args, parse_known=True)

    return ex_args

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