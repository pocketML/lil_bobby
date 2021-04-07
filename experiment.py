import os
from glob import glob
from sacred import Experiment
from sacred.observers import FileStorageObserver
from common.argparsers import args_experiment
from common import transponder
from finetune import main as finetune_main
from compress import main as compress_main
from evaluate import main as evaluate_main
from distill import main as distill_main

OUTPUT_DIR = "experiments"

TASK_FUNCS = {
    "finetune": finetune_main, "compress": compress_main,
    "evaluate": evaluate_main, "distill": distill_main
}

def run_experiment(task_args, _run):
    for task in task_args:
        TASK_FUNCS[task](task_args[task], sacred_experiment=_run)

if __name__ == "__main__":
    EXPERIMENT_ARGS, TASK_ARGS = args_experiment()

    EXPERIMENT = Experiment(EXPERIMENT_ARGS.name)

    if EXPERIMENT_ARGS.output_path is not None:
        OUTPUT_DIR = EXPERIMENT_ARGS.output_path

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    OUTPUT_STORAGE = FileStorageObserver(OUTPUT_DIR)

    EXPERIMENT.observers.append(OUTPUT_STORAGE)

    RUN_ID = EXPERIMENT_ARGS.name
    if os.path.exists(f"{OUTPUT_DIR}/{RUN_ID}"):
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
    elif "distill" in TASK_ARGS and TASK_ARGS["distill"].distill:
        DISTILLATION_ARGS = TASK_ARGS["distill"]
        max_epochs = DISTILLATION_ARGS.epochs
        transponder_args = dict(DISTILLATION_ARGS.__dict__)
        del transponder_args["epochs"]
        args_to_delete = ["size"]
        for arg in transponder_args:
            if transponder_args[arg] == False or transponder_args[arg] is None:
                args_to_delete.append(arg)
        for arg in args_to_delete:
            if arg in transponder_args:
                del transponder_args[arg]

    if transponder_args is not None:
        transponder.send_train_start(RUN_ID, transponder_args, max_epochs)

    RUN = EXPERIMENT._create_run("run_experiment", info={"name": RUN_ID})
    RUN._id = RUN_ID
    RUN()
