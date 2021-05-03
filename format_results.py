import json
from glob import glob

from common.argparsers import args_search

def experiment_contains_args(exp_path, exp_args, all_task_args):
    if exp_args.name is not None: # See if experiment matches name.
        with open(exp_path + "/info.json", "r", encoding="utf-8") as fp:
            if json.load(fp)["name"] == exp_args.name:
                return True
        return False

    # See if args used to run experiment matches given args.
    with open(exp_path + "/config.json", "r", encoding="utf-8") as fp:
        data = json.load(fp)["task_args"]
        for experiment_task in all_task_args:
            if experiment_task not in data:
                return False
            task_args = all_task_args[experiment_task]
            experiment_args = data[experiment_task]
            index = 0
            while index < len(task_args):
                key = task_args[index].replace("--", "").replace("-", "_")

                if key not in experiment_args:
                    return False

                value = True
                if not task_args[index+1].startswith("--"):
                    index += 1
                    value = task_args[index]

                if not isinstance(experiment_args[key], list):
                    experiment_args[key] = [experiment_args[key]]

                for experiment_value in experiment_args[key]:
                    if not str(experiment_value) == str(value):
                        return False

                index += 1
    return True

def find_matching_experiments(search_args, consumed_args):
    experiment_folders = glob("experiments/*")
    valid_folders = []

    for folder in experiment_folders:
        if "_sources" not in folder and experiment_contains_args(folder, search_args, consumed_args):
            valid_folders.append(folder)
    return valid_folders

def main(search_args, consumed_args):
    matching_folders = find_matching_experiments(search_args, consumed_args)
    print(matching_folders)

if __name__ == "__main__":
    SEARCH_ARGS, CONSUMED_ARGS = args_search()

    main(SEARCH_ARGS, CONSUMED_ARGS)
