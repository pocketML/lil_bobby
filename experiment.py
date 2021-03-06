import os
from sacred import Experiment
from sacred.observers import FileStorageObserver
from common.argparsers import args_experiment
from finetune import run_finetune
import compress

OUTPUT_DIR = "experiments"

def create_sacred_experiment(args):
    return Experiment(args.name)

def run_sacred_experiment(tasks_args):
    for task in tasks_args:
        run_finetune(tasks_args[task])

if __name__ == "__main__":
    EXPERIMENT_ARGS, TASK_ARGS = args_experiment()

    EXPERIMENT = create_sacred_experiment(EXPERIMENT_ARGS)
    EXPERIMENT.add_config(EXPERIMENT_ARGS.__dict__)

    if EXPERIMENT_ARGS.output_path is not None:
        OUTPUT_DIR = EXPERIMENT_ARGS.output_path

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    OUTPUT_STORAGE = FileStorageObserver(EXPERIMENT_ARGS.output)

    EXPERIMENT.observers.append(OUTPUT_STORAGE)

    EXPERIMENT.run()
