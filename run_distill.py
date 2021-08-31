from datetime import datetime

from common import argparsers
from compression.distillation.student_models.base import get_default_student_config
import run_all_seeds

def main(args, args_remain):
    # Create a list of arguments for use with 'run_all_seeds.py'.
    args_list = ["compress", "evaluate", "analyze", "--compression-actions", "distill"]
    args_list.extend(args_remain)

    # Add already specified args (task, alpha, student-arch, embed-type, embed-dim) to list.
    for key in args.__dict__:
        key_fmt = "--" + key.replace("_", "-")
        if key_fmt != "--only-original-data":
            args_list.append(key_fmt)
            args_list.append(str(args.__dict__[key]))
        elif args.only_original_data:
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
        args_list.extend(["--batch-size", "256"])

    args_list.extend(["--cls-hidden-dim", str(classifier_dim)])

    # Use 25% of our augmented data when running QQP or MNLI.
    if not args.only_original_data and large_task:
        args_list.extend(["--bootstrap-data-ratio", "0.25"])

    # Edit 'submit.job' to set minimum required memory.
    lines = []
    with open("submit.job", "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    mem_per_cpu = 6000

    with open("submit.job", "w", encoding="utf-8") as fp:
        for line in lines:
            new_line = line
            if "#SBATCH --mem-per-cpu=" in line:
                if large_task:
                    new_line = f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
                else:
                    new_line = f"##SBATCH --mem-per-cpu={mem_per_cpu}\n"
            elif "#SBATCH --time=" in line:
                time = "12:00:00"
                if args.student_arch == "emb-ffn":
                    time = "04:00:00"
                elif args.embedding_type == "char":
                    time = "32:00:00"
                new_line = f"#SBATCH --time={time}\n"

            fp.write(new_line)

    # Include final static arguments.
    args_list.extend([
        "--embedding-freeze", "False", "--model-size",
        "--model-disk-size", "--transponder"
    ])

    # Create a name for the experiment that we are running.
    alpha_fmt = "05" if args.alpha == 0.5 else str(int(args.alpha))
    dt_now = datetime.now()
    month_names = ["may", "june", "july", "august"]
    date_fmt = f"{month_names[dt_now.month - 5]}{dt_now.day}"
    arch_fmt = args.student_arch if args.student_arch != "emb-ffn" else "embffn"
    task_fmt = args.task.replace("sst-2", "sst")
    name = f"{arch_fmt}_{task_fmt}_alpha{alpha_fmt}_{args.embedding_type}{args.embedding_dim}"
    if args.only_original_data:
        name += "_og"
    name += f"_{date_fmt}"
    args_list.extend(["--name", name])

    final_args, args_remain = argparsers.args_run_all(args_list)

    run_all_seeds.main(final_args, args_remain)

if __name__ == "__main__":
    ARGS, ARGS_REMAIN = argparsers.args_run_distill()

    main(ARGS, ARGS_REMAIN)
