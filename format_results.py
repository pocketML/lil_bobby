import json
from glob import glob
from datetime import datetime

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

def experiment_contains_args(exp_path, meta_args, search_args):
    if hasattr(search_args, "name"): # See if experiment matches name.
        with open(exp_path + "/info.json", "r", encoding="utf-8") as fp:
            if json.load(fp)["name"] == search_args.name:
                return True
        return False

    # See if args used to run experiment matches given args.
    # We search for both cmd args given to experiment as well as model cfg args used.
    values_found = {x: False for x in search_args.__dict__}
    column_value = None
    row_value = None
    for data_type in ("config", "model_cfg"):
        experiment_args = get_json_data(exp_path, data_type)

        if experiment_args is None:
            continue

        if data_type == "config":
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
                for value in values:
                    if str(experiment_value) == value:
                        value_found = True
                        if key == meta_args.table_col:
                            column_value = value
                        elif key == meta_args.table_row:
                            row_value = value
                        break
                if value_found:
                    break

            values_found[key] = value_found

    return all(values_found.values()), column_value, row_value

def get_experiment_date(folder):
    with open(folder + "/run.json", "r", encoding="utf-8") as fp:
        data = json.load(fp)["stop_time"]
        dash_split = data.split("-")
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

def find_matching_experiments(search_args, meta_args):
    experiment_folders = list(filter(lambda x: "_sources" not in x, glob("experiments/*")))

    experiment_folders.sort(key=get_experiment_date, reverse=True)

    data_dict = {x: [] for x in search_args.__dict__[meta_args.table_row]}

    for folder in experiment_folders:
        experiment_matches, col_value, row_value = experiment_contains_args(folder, meta_args, search_args)
        if experiment_matches:
            with open(folder + "/metrics.json", "r", encoding="utf-8") as fp:
                metrics_data = json.load(fp)
                data = None
                if meta_args.table_metric == "accuracy":
                    if "test.accuracy" in metrics_data:
                        accuracy = metrics_data["test.accuracy"]["values"][0]
                    else:
                        accuracy = max(metrics_data["validation.acc"]["values"])
                    data = accuracy

                if "model_params" in metrics_data and meta_args.table_metric == "params":
                    data = metrics_data["model_params"]["values"][0]

                if "model_size" in metrics_data and meta_args.table_metric == "size":
                    data = metrics_data["model_size"]["values"][0]

                if len(data_dict[row_value]) < len(search_args.__dict__[meta_args.table_col]):
                    data_dict[row_value].append((col_value, data))

    ordered_data = [(k, v) for k, v in data_dict.items()]

    ordered_data.sort(key=lambda x: search_args.__dict__[meta_args.table_row].index(x[0]))
    for row_key, data in ordered_data:
        data.sort(key=lambda x: search_args.__dict__[meta_args.table_col].index(x[0]))

    return ordered_data

def main(meta_args, search_args):
    row_data = find_matching_experiments(search_args, meta_args)
    if meta_args.generate_table:
        with open("test.txt", "w", encoding="utf-8") as fp:
            if meta_args.table_headers is not None:
                fp.write(" & " + " & ".join(meta_args.table_headers) + "\n")
            else:
                fp.write(" & " + " & ".join(search_args.__dict__[meta_args.table_col]) + "\n")

            for row_key, row_data in row_data:
                line = f"{row_key}"
                for col, data in row_data:
                    line += f" & {data}"
                fp.write(line + "\n")

if __name__ == "__main__":
    META_ARGS, SEARCH_ARGS = args_search()

    main(META_ARGS, SEARCH_ARGS)
