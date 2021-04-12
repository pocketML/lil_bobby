import argparse
from common.task_utils import TASK_INFO
from common.model_utils import MODEL_INFO
from compression.distillation.models import STUDENT_MODELS

FINETUNE_TASKS = list(TASK_INFO.keys())
MODEL_ARCHS = list(MODEL_INFO.keys()) + list(STUDENT_MODELS.keys())

# define arguments for knowledge distillation
def args_distill(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    student_archs = ["glue", "wasserblat-ffn", "tang"]

    ap.add_argument("--task", choices=FINETUNE_TASKS, required=True)
    ap.add_argument("--checkpoint-path", default="checkpoints")
    ap.add_argument("--student-arch", type=str, choices=student_archs, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=int, default=1)
    ap.add_argument("--train-cbow", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--loadbar", action="store_true")
    ap.add_argument("--eval", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--early-stopping", type=int, default=5)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

def args_prune(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruning-threshold", type=float)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

# define arguments for model compression
def args_compress():
    ap = argparse.ArgumentParser()
    compression_techniques = [
        'distill',
        'prune-magnitude',
        'prune-movement',
        'quantize-dynamic',
        'quantize-static',
        'quantize-aware'
    ]
    ap.add_argument("--distill", action="store_true")

    compression_args, args_remain = ap.parse_known_args()
    if compression_args.distill:
        compression_args, args_remain = args_finetune(
            args_remain, namespace=compression_args, parse_known=True
        )

    return compression_args

def args_cbow(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=FINETUNE_TASKS, required=True)
    ap.add_argument("--context-size", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

def args_preprocess(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    augmenters = ["tinybert", "masked", "pos", "ngram"]
    teacher_archs = ["roberta_large", "roberta_base"]

    ap.add_argument("--glue-preprocess", action="store_true")
    ap.add_argument("--augment", type=str, choices=augmenters, default=None)
    ap.add_argument("--generate-loss", type=str, choices=("processed", "tinybert"), default=None)
    ap.add_argument("--teacher-arch", choices=teacher_archs, default="roberta_large")
    ap.add_argument("--task", choices=FINETUNE_TASKS, required=True)
    ap.add_argument("--seed", type=int, default=1337)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

def args_evaluate(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument("--loadbar", action="store_true")
    ap.add_argument('--arch', choices=MODEL_ARCHS, required=True)

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
    ap = argparse.ArgumentParser()

    ap.add_argument('--model-name', type=str, default=None)
    ap.add_argument('--arch', choices=MODEL_ARCHS, required=True)
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

    return experiment_args, task_args
