from common import argparsers, task_utils
from os import system

def main(args, args_remain):
    name_1 = args.name
    name_2 = args.model_name
    if name_1 is None and name_2 is None:
        print("Both --name or --model-name can't be None.")
        return
    args_str = " ".join(args_remain)

    if args.seed_names == []:
        seed_names = task_utils.SEED_DICT.keys()
    else:
        seed_names = args.seed_names

    for seed_name in seed_names:
        full_cmd_str = f"sbatch submit.job experiment.py {args_str}"
        if name_1 is not None:
            full_cmd_str += f" --name {name_1}_{seed_name} --seed {task_utils.SEED_DICT[seed_name]}"
        if name_2 is not None:
            full_cmd_str += f" --model-name {name_1}_{seed_name}"

        system(full_cmd_str)

if __name__ == "__main__":
    ARGS, ARGS_REMAIN = argparsers.args_run_all()

    main(ARGS, ARGS_REMAIN)
