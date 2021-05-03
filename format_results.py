import json
from glob import glob
from sys import meta_path

from common.argparsers import args_search

def get_json_data(experiment_path, data_type):
    try:
        with open(f"{experiment_path}/{data_type}.json", "r", encoding="utf-8") as fp:
            data = json.load(fp)
            if data_type == "config":
                data = data["task_args"]
            return data
    except FileNotFoundError:
        return None

def experiment_contains_args(exp_path, job, search_args):
    if hasattr(search_args, "name"): # See if experiment matches name.
        with open(exp_path + "/info.json", "r", encoding="utf-8") as fp:
            if json.load(fp)["name"] == search_args.name:
                return True
        return False

    # See if args used to run experiment matches given args.
    # We search for both cmd args given to experiment as well as model cfg args used.
    values_found = {x: False for x in search_args.__dict__}
    for data_type in ("config", "model_cfg"):
        experiment_args = get_json_data(exp_path, data_type)
        if data_type == "config":
            if job not in experiment_args:
                return False

            experiment_args = experiment_args[job]

        if experiment_args is None:
            continue

        for key in search_args.__dict__:
            if values_found[key]:
                continue

            values = search_args.__dict__[key]

            if key not in experiment_args:
                continue

            if not isinstance(experiment_args[key], list):
                experiment_args[key] = [experiment_args[key]]

            value_found = False

            for experiment_value in experiment_args[key]:
                for value in values:
                    if str(experiment_value) == value:
                        value_found = True
                        break
                if value_found:
                    break

            values_found[key] = value_found

    return all(values_found.values())

def find_matching_experiments(search_args, job):
    experiment_folders = glob("experiments/*")
    valid_folders = []

    for folder in experiment_folders:
        if "_sources" not in folder and experiment_contains_args(folder, job, search_args):
            valid_folders.append(folder)
    return valid_folders

def main(meta_args, search_args):
    matching_folders = find_matching_experiments(search_args, meta_args.job)
    print(matching_folders)

if __name__ == "__main__":
    META_ARGS, SEARCH_ARGS = args_search()

    main(META_ARGS, SEARCH_ARGS)
