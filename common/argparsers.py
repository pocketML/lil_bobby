import argparse
from common.task_utils import TASK_INFO, SEED_DICT
from common.model_utils import MODEL_INFO
from embedding.embeddings import EMBEDDING_ZOO
from compression.distillation.student_models.base import get_default_student_config
from compression.distillation.models import STUDENT_MODELS

FINETUNE_TASKS = list(TASK_INFO.keys())
MODEL_ARCHS = list(MODEL_INFO.keys()) + list(STUDENT_MODELS.keys())

# define arguments for knowledge distillation
def args_distill(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--temperature", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0)
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
    ap.add_argument("--prune-magnitude", action="store_true")
    ap.add_argument("--prune-topk", action="store_true")
    ap.add_argument("--prune-movement", action="store_true")
    ap.add_argument("--prune-aware", action="store_true")
    ap.add_argument("--prune-warmup", type=int, default=5)
    ap.add_argument("--prune-threshold", type=float, required=True)

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# define arguments for model compression
def args_compress(args=None, namespace=None, parse_known=False):
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
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--loadbar", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seed-name", type=str, choices=SEED_DICT.keys(), default=None)

    compression_args, args_remain = ap.parse_known_args(args=args, namespace=namespace)

    if compression_args.seed_name is not None:
        setattr(compression_args, "seed", SEED_DICT[compression_args.seed_name])

    overwrite_ap = argparse.ArgumentParser()
    cfg = get_default_student_config(compression_args.task, compression_args.student_arch)
    for key, value in cfg.items():
        if isinstance(value, bool):
            overwrite_ap.add_argument("--" + key, type=str2bool, nargs="?", const=True, default=value)
        else:
            overwrite_ap.add_argument("--" + key, type=type(value))

    compression_args, args_remain = overwrite_ap.parse_known_args(args_remain, namespace=compression_args)

    for action in compression_args.compression_actions:
        compression_args, args_remain = compression_actions[action](
            args_remain, namespace=compression_args, parse_known=True
        )

    return compression_args, args_remain

def args_preprocess(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    augmenters = ["tinybert", "masked", "pos", "ngram"]
    teacher_archs = ["roberta_large", "roberta_base"]

    ap.add_argument("--glue-preprocess", action="store_true")
    ap.add_argument("--augment", type=str, choices=augmenters, default=None)
    ap.add_argument("--generate-loss", type=str, choices=("processed", "tinybert"), default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--teacher-arch", choices=teacher_archs, default="roberta_large")
    ap.add_argument("--task", choices=FINETUNE_TASKS, required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--loadbar", action="store_true")

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)
    return ap.parse_args(args=args, namespace=namespace)

def args_embeddings(args=None, namespace=None, parse_known=False):
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", choices=TASK_INFO.keys(), required=True)
    ap.add_argument("--embed-type", type=str, choices=EMBEDDING_ZOO.keys(), required=True)
    ap.add_argument("--context-size", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--original-data", action="store_true")
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

def args_analyze(args=None, namespace=None, parse_known=False):
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

    if parse_known:
        return ap.parse_known_args(args=args, namespace=namespace)

    return ap.parse_args(args=args, namespace=namespace)

def args_experiment():
    ap = argparse.ArgumentParser()

    task_choices = ("finetune", "compress", "evaluate", "analyze")
    ap.add_argument("jobs", nargs="+", choices=task_choices)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--transponder", action="store_true")
    ap.add_argument("--output-path", type=str)

    task_args = {}

    experiment_args, args_remain = ap.parse_known_args()

    # Fill in some missing arguments, that we can automatically infer, for convenience sake.
    if "evaluate" in experiment_args.jobs or "analyze" in experiment_args.jobs:
        if "--model-name" not in args_remain: # We are missing --model-name (copy --name).
            args_remain.extend(["--model-name", experiment_args.name])
        if ("--arch" in args_remain) ^ ("--student-arch" in args_remain):
            # We are missing either --arch or --student-arch.
            if "--arch" in args_remain: # Copy --arch value to --student-arch.
                key = "--student-arch"
                value = args_remain[args_remain.index("--arch") + 1]
            else: # Copy --student-arch value to --arch.
                key = "--arch"
                value = args_remain[args_remain.index("--student-arch") + 1]

            args_remain.extend([key, value])

    argparse_funcs = {
        "finetune": args_finetune, "compress": args_compress,
        "evaluate": args_evaluate, "analyze": args_analyze
    }

    for task in experiment_args.jobs:
        args_for_task = argparse_funcs[task](args_remain, parse_known=True)[0]
        task_args[task] = args_for_task

    return experiment_args, task_args

def args_search():
    usage = """
    Search for any 
    """
    ap = argparse.ArgumentParser()

    # Job to search for.
    task_choices = ("finetune", "compress", "evaluate", "analyze")
    ap.add_argument("job", choices=task_choices)

    # Args for manipulating found results.
    ap.add_argument("--metric", type=str, choices=("acc", "params", "size"), default="acc")
    ap.add_argument("--sort", choices=("time", "name"), default="time")
    ap.add_argument("--tab-separate", action="store_true")
    ap.add_argument("--generate-table", action="store_true")
    ap.add_argument("--table-col", type=str)
    ap.add_argument("--table-row", type=str)
    ap.add_argument("--table-headers", nargs="+")

    meta_args, args_remain = ap.parse_known_args()

    search_ap = argparse.ArgumentParser()

    index = 0
    # Go through remaining arguments and add them as search keys/args to "search_ap".
    while index < len(args_remain):
        arg = args_remain[index]
        if arg.startswith("--"):
            if index == len(args_remain) + 1:
                print(f"Error: Missing value for argument {arg}")
                exit(0)

            search_ap.add_argument(arg, type=str, nargs="+")
        index += 1

    search_args = search_ap.parse_args(args_remain)

    return meta_args, search_args

def args_run_all():
    ap = argparse.ArgumentParser()

    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--seed-names", type=str, nargs="+", choices=SEED_DICT.keys())

    return ap.parse_known_args()
