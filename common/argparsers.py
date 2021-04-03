import argparse
from fairseq import options
from common.task_utils import TASK_INFO
from common.model_utils import MODEL_INFO
from compression.distillation.models import STUDENT_MODELS

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

def args_cbow(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=finetune_tasks, required=True)
    ap.add_argument("--context-size", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

# define arguments for knowledge distillation
def args_distill(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    augmenters = ["tinybert", "masked", "pos", "ngram"]
    student_archs = ["glue", "wasserblat-ffn", "tang"]
    teacher_archs = ["roberta_large", "roberta_base"]

    ap.add_argument("--task", choices=finetune_tasks, required=True)
    ap.add_argument("--teacher-arch", choices=teacher_archs, default="roberta_large")
    ap.add_argument("--checkpoint-path", default="checkpoints")
    ap.add_argument("--generate-loss", type=str, choices=("processed", "tinybert"))
    ap.add_argument("--student-arch", type=str, choices=student_archs, default="glue")
    ap.add_argument("--augment", type=str, choices=augmenters)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=int, default=1)
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--train-cbow", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--size", action="store_true")

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
    model_archs = list(MODEL_INFO.keys()) + list(STUDENT_MODELS.keys())
    ap = argparse.ArgumentParser()

    ap.add_argument('--model-name', type=str, default=None)
    ap.add_argument('--arch', choices=model_archs, required=True)
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

    task_choices = ("finetune", "compress", "evaluate", "analyze", "distill")
    ap.add_argument("jobs", nargs="+", choices=task_choices)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--transponder", action="store_true")
    ap.add_argument("--output-path", type=str)

    task_args = {}

    experiment_args, args_remain = ap.parse_known_args()
    for task in experiment_args.jobs:
        if task == "finetune":
            finetune_args = args_finetune(args_remain, parse_known=True)[0]
            task_args["finetune"] = finetune_args
        if task == "compress":
            compress_args = args_compress(args_remain, parse_known=True)[0]
            task_args["compress"] = compress_args
        if task == "evaluate":
            evaluate_args = args_evaluate(args_remain, parse_known=True)[0]
            task_args["evaluate"] = evaluate_args
        if task == "distill":
            distill_args = args_distill(args_remain, parse_known=True)[0]
            task_args["distill"] = distill_args

    return experiment_args, task_args
