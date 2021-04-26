import argparse
from common.task_utils import TASK_INFO, SEED_DICT
from common.model_utils import MODEL_INFO
from compression.distillation.models import STUDENT_MODELS

FINETUNE_TASKS = list(TASK_INFO.keys())
MODEL_ARCHS = list(MODEL_INFO.keys()) + list(STUDENT_MODELS.keys())

# define arguments for knowledge distillation
def args_distill(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--temperature", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--early-stopping", type=int, default=10)
    ap.add_argument("--original-data", action="store_true")

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

def args_quantize(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptq-embedding", action="store_true")
    ap.add_argument("--dq-encoder", action="store_true")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dq-classifier", action="store_true")
    group.add_argument("--ptq-classifier", action="store_true")

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

def args_prune(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune-magnitude-static", action="store_true")
    ap.add_argument("--prune-magnitude-aware", action="store_true")
    ap.add_argument("--prune-movement", action="store_true")
    ap.add_argument("--prune-threshold", type=float, required=True)
    ap.add_argument("--prune-compute", type=float, default=15)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

# define arguments for model compression
def args_compress():
    ap = argparse.ArgumentParser()
    compression_actions = {
        'distill': args_distill,
        'prune': args_prune,
        'quantize': args_quantize,
    }
    ap.add_argument("--compression-actions", nargs="+", choices=compression_actions.keys(), required=True)

    ap.add_argument("--task", choices=FINETUNE_TASKS, required=True)
    ap.add_argument("--load-trained-model", type=str)
    ap.add_argument("--student-arch", type=str, choices=STUDENT_MODELS.keys(), required=True)
    ap.add_argument("--checkpoint-path", default="checkpoints")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--loadbar", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seed-name", type=str, choices=SEED_DICT.keys(), default=None)

    compression_args, args_remain = ap.parse_known_args()

    for action in compression_args.compression_actions:
        compression_args, args_remain = compression_actions[action](
            args_remain, namespace=compression_args, parse_known=True
        )

    return compression_args, args_remain

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
    ap.add_argument("--checkpoint-path", default="checkpoints")
    ap.add_argument("--teacher-arch", choices=teacher_archs, default="roberta_large")
    ap.add_argument("--task", choices=FINETUNE_TASKS, required=True)
    ap.add_argument("--cpu", action="store_true")
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
    ap.add_argument('--model-disk-size', action="store_true")

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
            compress_args = args_compress()[0]
            task_args["compress"] = compress_args
        if task == "evaluate":
            evaluate_args = args_evaluate(args_remain, parse_known=True)[0]
            task_args["evaluate"] = evaluate_args

    return experiment_args, task_args

def args_run_all():
    ap = argparse.ArgumentParser()

    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--model-name", type=str, default=None)

    return ap.parse_known_args()
