import os
import json
from re import A
from shutil import rmtree
from glob import glob
from numpy.lib.arraysetops import isin
from sacred import Experiment
from sacred.observers import FileStorageObserver
from common.argparsers import args_experiment
from common import transponder
from finetune import main as finetune_main
from compress import main as compress_main
from evaluate import main as evaluate_main
from analyze import main as analyze_main

OUTPUT_DIR = "experiments"

TASK_FUNCS = {
    "finetune": finetune_main, "compress": compress_main,
    "evaluate": evaluate_main, "analyze": analyze_main
}

def run_experiment(task_args, _run):
    for task in task_args:
        TASK_FUNCS[task](task_args[task], sacred_experiment=_run)

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

def search_for_experiments(exp_args, task_args):
    experiment_folders = glob("experiments/*")
    valid_folders = []

    for folder in experiment_folders:
        if "_sources" not in folder and experiment_contains_args(folder, exp_args, task_args):
            valid_folders.append(folder)
    return valid_folders

if __name__ == "__main__":
    EXPERIMENT_ARGS, TASK_ARGS, CONSUMED_ARGS = args_experiment()
    if EXPERIMENT_ARGS.search:
        experiments = search_for_experiments(EXPERIMENT_ARGS, CONSUMED_ARGS)
        print(experiments)
        exit(0)
    elif EXPERIMENT_ARGS.name is None:
        print("Name of experiment must be provided (unless --search is used).")
        exit(0)

    EXPERIMENT = Experiment(EXPERIMENT_ARGS.name)

    if EXPERIMENT_ARGS.output_path is not None:
        OUTPUT_DIR = EXPERIMENT_ARGS.output_path

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_STORAGE = FileStorageObserver(OUTPUT_DIR)

    EXPERIMENT.observers.append(OUTPUT_STORAGE)

    RUN_ID = EXPERIMENT_ARGS.name
    if os.path.exists(f"{OUTPUT_DIR}/{RUN_ID}"):
        if EXPERIMENT_ARGS.overwrite:
            rmtree(f"{OUTPUT_DIR}/{RUN_ID}")
        else:
            PREVIOUS_RUN = glob(f"{OUTPUT_DIR}/{RUN_ID}*")
            INDEX = PREVIOUS_RUN[-1].split("_")[-1]
            try:
                INDEX = int(INDEX) + 1
            except ValueError:
                INDEX = 2
            RUN_ID = f"{RUN_ID}_{INDEX}"

    EXPERIMENT.add_config({
        "task_args": TASK_ARGS
    })
    EXPERIMENT.command(run_experiment)

    transponder.TRANSPONDER_ACTIVE = EXPERIMENT_ARGS.transponder

    transponder_args = None
    max_epochs = None

    if "finetune" in TASK_ARGS:
        FINETUNE_ARGS = TASK_ARGS["finetune"]
        max_epochs = FINETUNE_ARGS.max_epochs
        transponder_args = dict(FINETUNE_ARGS.__dict__)
        del transponder_args["max_epochs"]
    elif "compress" in TASK_ARGS and "distill" in TASK_ARGS["compress"].compression_actions:
        DISTILLATION_ARGS = TASK_ARGS["compress"]
        max_epochs = DISTILLATION_ARGS.epochs
        transponder_args = dict(DISTILLATION_ARGS.__dict__)
        del transponder_args["epochs"]
        del transponder_args["compression_actions"]
        args_to_delete = ["size"]
        for arg in transponder_args:
            if transponder_args[arg] is False or transponder_args[arg] is None:
                args_to_delete.append(arg)
        for arg in args_to_delete:
            if arg in transponder_args:
                del transponder_args[arg]

    if transponder_args is not None:
        transponder.send_train_start(RUN_ID, transponder_args, max_epochs)

    RUN = EXPERIMENT._create_run("run_experiment", info={"name": RUN_ID})
    RUN._id = RUN_ID
    RUN()
