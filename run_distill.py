from datetime import datetime

from common import argparsers
from compression.distillation.student_models.base import get_default_student_config
import run_all_seeds

def main(args):
    # Create a list of arguments for use with 'run_all_seeds.py'.
    args_list = ["compress", "evaluate", "analyze", "--compression-actions", "distill"]

    # Add already specified args (task, alpha, student-arch, embed-type, embed-dim) to list.
    for key in args.__dict__:
        key_fmt = "--" + key.replace("_", "-")
        if key_fmt != "--original-data":
            args_list.append(key_fmt)
            args_list.append(str(args.__dict__[key]))
        elif args.original_data:
            args_list.append(key_fmt)

    # Load default config for the given task and student.
    task_cfg = get_default_student_config(args.task, args.student_arch)

    large_task = args.task in ("qqp", "mnli")

    if args.student_arch != "emb-ffn":
        # Specify encoder hidden dimension, if not running FFN student.
        encoder_dim = task_cfg["encoder-hidden-dim"]
        if large_task: # Double encoder dim on QQP and MNLI tasks.
            encoder_dim = encoder_dim * 2
        args_list.extend(["--encoder-hidden-dim", str(encoder_dim)])

    # Specify classifier hidden dimension.
    classifier_dim = task_cfg["cls-hidden-dim"]
    if large_task: # Double classifier dim on QQP and MNLI tasks.
        classifier_dim = classifier_dim * 2

    args_list.extend(["--cls-hidden-dim", str(classifier_dim)])

    if not args.original_data and large_task:
        args_list.extend(["--data-ratio", "0.25"])

    args_list.extend([
         "--embedding-freeze", "False", "--vocab-size", "5000", "--epochs", "50",
        "--model-size", "--model-disk-size", "--transponder"
    ])

    # Create a name for the experiment that we are running.
    alpha_fmt = "05" if args.alpha == 0.5 else str(int(args.alpha))
    dt_now = datetime.now()
    month_names = ["may", "june", "july", "august"]
    date_fmt = f"{month_names[dt_now.month - 5]}{dt_now.day}"
    task_fmt = args.task.replace("sst-2", "sst")
    name = f"{args.student_arch}_{task_fmt}_alpha{alpha_fmt}_{args.embedding_type}{args.embedding_dim}_{date_fmt}"
    args_list.extend(["--name", name])

    final_args, args_remain = argparsers.args_run_all(args_list)

    run_all_seeds.main(final_args, args_remain)

if __name__ == "__main__":
    ARGS = argparsers.args_run_experiment()

    main(ARGS)
