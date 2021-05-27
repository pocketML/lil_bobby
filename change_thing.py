from glob import glob
import json

all_experiment_files = glob("experiments/*")

for experiment_folder in all_experiment_files:
    cfg_file = f"{experiment_folder}/config.json"
    try:
        with open(cfg_file, "r") as fp:
            config = json.load(fp)

        if "task_args" in config and "compress" in config["task_args"]:
            parameters = config["task_args"]["compress"]

            if "data_ratio" in parameters:
                parameters["bootstrap_data_ratio"] = parameters["data_ratio"]
                del parameters["data_ratio"]

        with open(cfg_file, "w") as fp:
            json.dump(config, fp)
    except FileNotFoundError:
        pass
