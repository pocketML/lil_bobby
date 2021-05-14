import json
from glob import glob
from datetime import datetime
import numpy as np

from common.task_utils import SEED_DICT
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

def value_matches(key, search_value, value):
    if key == "name":
        return search_value in value
    if isinstance(value, bool):
        return (search_value == "True") == value
    return type(value)(search_value) == value

def get_experiment_date(time_str):
    dash_split = time_str.split("-")
    year = int(dash_split[0])
    month = int(dash_split[1])
    colon_split = dash_split[2].split(":")
    t_split = colon_split[0].split("T")
    day = int(t_split[0])
    hour = int(t_split[1])
    minute = int(colon_split[1])
    dot_split = colon_split[2].split(".")
    second = int(dot_split[0])
    return datetime(year, month, day, hour, minute, second).timestamp()

def experiment_contains_args(exp_path, meta_args, search_args):
    # See if args used to run experiment matches given args.
    # We search for both cmd args given to experiment as well as model cfg args used.
    values_found = {x: False for x in search_args.__dict__}
    data_found = {}

    for data_type in ("info", "config", "model_cfg"):
        experiment_args = get_json_data(exp_path, data_type)

        if experiment_args is None:
            continue

        if data_type == "info" and "stop_time" in experiment_args:
            data_found["timestamp"] = get_experiment_date(experiment_args["stop_time"])
        elif data_type == "config":
            if meta_args.job not in experiment_args:
                break

            experiment_args = experiment_args[meta_args.job]

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
                for search_value in values:
                    if value_matches(key, search_value, experiment_value):
                        value_found = True
                        data_found[key] = search_value
                        break
                if value_found:
                    break

            values_found[key] = value_found

    if not all(values_found.values()):
        return None

    name = exp_path.replace("\\", "/").split("/")[-1]
    data_found["name"] = name
    return data_found

def get_seed_index(name):
    seed_list = list(SEED_DICT)

    for index, seed_name in enumerate(seed_list):
        if seed_name in name:
            return index
    return 0

def find_matching_experiments(meta_args, search_args):
    experiment_folders = list(filter(lambda x: "_sources" not in x, glob("experiments/*")))

    data = []

    for folder in experiment_folders:
        experiment_data = experiment_contains_args(folder, meta_args, search_args)
        if experiment_data is not None:
            with open(folder + "/metrics.json", "r", encoding="utf-8") as fp:
                metrics_data = json.load(fp)

                accuracy = None
                if "test.accuracy" in metrics_data:
                    accuracy = metrics_data["test.accuracy"]["values"][0]
                elif "validation.acc" in metrics_data:
                    accuracy = max(metrics_data["validation.acc"]["values"])

                if accuracy is not None:
                    experiment_data["acc"] = f"{accuracy:.4f}"

                if "model_params" in metrics_data:
                    experiment_data["params"] = metrics_data["model_params"]["values"][0]

                if "model_size" in metrics_data:
                    experiment_data["size"] = f"{metrics_data['model_size']['values'][0]:.3f}"

            data.append(experiment_data)

    if meta_args.sort == "time":
        data.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    elif meta_args.sort == "name":
        data.sort(key=lambda x: get_seed_index(x["name"]), reverse=False)

    return data

def format_found_data(found_data, meta_args, search_args):
    data_dict = {x: [] for x in search_args.__dict__[meta_args.table_row]}

    for experiment_data in found_data:
        row_value = str(experiment_data[meta_args.table_row])
        col_value = str(experiment_data[meta_args.table_col])
        if not meta_args.metric in experiment_data:
            raise KeyError(f"One of the experiments is missing the metric '{meta_args.metric}'")

        result = experiment_data[meta_args.metric]
        if len(data_dict[row_value]) < len(search_args.__dict__[meta_args.table_col]):
            data_dict[row_value].append((col_value, result))

    ordered_data = [(k, v) for k, v in data_dict.items()]

    ordered_data.sort(key=lambda x: search_args.__dict__[meta_args.table_row].index(x[0]))
    for _, data in ordered_data:
        data.sort(key=lambda x: search_args.__dict__[meta_args.table_col].index(x[0]))

    return ordered_data

def main(meta_args, search_args):
    found_data = find_matching_experiments(meta_args, search_args)
    if meta_args.generate_table:
        necessary_attr = ("metric", "table_col", "table_row")
        for attr in necessary_attr:
            if meta_args.__dict__[attr] is None:
                print(f"Generate table: Missing necessary argument '--{attr}'")
                exit(0)

        ordered_data = format_found_data(found_data, meta_args, search_args)

        if meta_args.table_headers is not None:
            print(" & " + " & ".join(meta_args.table_headers))
        else:
            line = ""
            for value in search_args.__dict__[meta_args.table_col]:
                line += f" & {meta_args.table_col}={value}"
            print(line)

        for row_key, row_data in ordered_data:
            line = f"{meta_args.table_row}={row_key}"
            for col, data in row_data:
                line += f" & {data}"
            print(line)
    else:
        if found_data != []:
            accuracies = np.array([float(data_point["acc"]) for data_point in found_data])
            mean = np.mean(accuracies)
            std_dev = np.std(accuracies)

        sort_order = ["acc", "mean", "std_dev", "params", "size"]

        for index, data_point in enumerate(found_data):
            if meta_args.tab_separate:
                actual_sort_order = sort_order
                data_point["mean"] = mean if index == 0 else " "
                data_point["std_dev"] = std_dev if index == 0 else " "
                line = "\t".join(str(data_point[x]) for x in actual_sort_order)
            else:
                if index == 0:
                    data_point["mean"] = mean
                    data_point["std_dev"] = std_dev
                line = ", ".join([f"{k}={v}" for (k, v) in data_point.items()])
            print(line)

if __name__ == "__main__":
    META_ARGS, SEARCH_ARGS = args_search()

    main(META_ARGS, SEARCH_ARGS)
