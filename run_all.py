from common import argparsers, task_utils
from os import system

def main(args, args_remain):
    name_1 = args.name
    name_2 = args.model_name
    if name_1 is None and name_2 is None:
        print("Both --name or --model-name can't be None.")
        return
    args_str = " ".join(args_remain)

    for seed_name in task_utils.SEED_DICT:
        full_cmd_str = f"python experiment.py {args_str}"
        if name_1 is not None:
            full_cmd_str += f" --name {name_1}_{seed_name} --seed {task_utils.SEED_DICT[seed_name]}"
        if name_2 is not None:
            full_cmd_str += f" --model-name {name_1}_{seed_name}"

        system(full_cmd_str)

if __name__ == "__main__":
    ARGS, ARGS_REMAIN = argparsers.args_run_all()

    main(ARGS, ARGS_REMAIN)
