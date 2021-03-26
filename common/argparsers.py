import argparse
from fairseq import options
from common.task_utils import TASK_INFO
from common.model_utils import MODEL_INFO

finetune_tasks = list(TASK_INFO.keys())

# define arguments for model compression
def args_compress(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    compression_techniques = [
        'prune-magnitude',
        'quantize'
    ]
    ap.add_argument("--techniques", choices=compression_techniques, nargs="+", required=True)
    ap.add_argument("--pruning-threshold", type=float)
    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

# define arguments for knowledge distillation
def args_distill(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", choices=finetune_tasks, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--teacher-arch", choices=["roberta_large", "roberta_base"], )
    ap.add_argument("--distillation", action="store_true")
    ap.add_argument("--generate-loss", action="store_true")
    augmenters = ["tinybert", "masked", "pos", "ngram"]
    students = ["glue", "wasserblat-ffn", "tang"]
    ap.add_argument("--student-arch", type=str, choices=students, default="glue")
    ap.add_argument("--augment", type=str, choices=augmenters)
    ap.add_argument("--augment2", type=str, choices=augmenters)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--size", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

# download benchmark tasks, roberta models etc
def args_download():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", "-t", choices=list(TASK_INFO.keys()) + ["glue"])
    group.add_argument("--model", "-m", choices=MODEL_INFO.keys())
    return ap.parse_args()

def args_evaluate(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument('--cpu', action='store_true')

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

def args_finetune(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    arch_choices = ['roberta_base', 'roberta_large']

    ap.add_argument("--task", "-t", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--arch", choices=arch_choices, default="roberta_large")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--model-parallel-size", type=int, default=1)
    ap.add_argument("--cpu", action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--seed', type=int, default=233) # 233 gave ~6 percentage points higher accuracy on RTE that seed=1

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

def args_analyze():
    models = list(MODEL_INFO.keys())
    ap = argparse.ArgumentParser()

    ap.add_argument('--model-name', type=str, required=True)
    ap.add_argument('--task', choices=TASK_INFO.keys(), required=True)
    ap.add_argument('--model-size', action='store_true')
    ap.add_argument('--weight-hist', action='store_true')
    ap.add_argument('--layer-weight-hist', type=str)
    ap.add_argument('--named-params', action='store_true')
    ap.add_argument('--weight-thresholds', action='store_true')

    args = ap.parse_args()
    return args

def args_experiment():
    ap = argparse.ArgumentParser()

    task_choices = ("finetune", "compress", "evaluate", "analyze")
    ap.add_argument("tasks", nargs="+", choices=task_choices)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--transponder", action="store_true")
    ap.add_argument("--output-path", type=str)

    task_args = {}

    experiment_args, args_remain = ap.parse_known_args()
    for task in experiment_args.tasks:
        if task == "finetune":
            finetune_args, args_remain = args_finetune(args_remain, parse_known=True)
            task_args["finetune"] = finetune_args
        if task == "compress":
            compress_args, args_remain = args_compress(args_remain, parse_known=True)
            task_args["compress"] = compress_args
        if task == "evaluate":
            evaluate_args, args_remain = args_evaluate(args_remain, parse_known=True)
            task_args["evaluate"] = evaluate_args

    return experiment_args, task_args
