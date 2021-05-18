import os
from sys import stdout
from shutil import rmtree
from glob import glob
from sacred import Experiment
from sacred.observers import FileStorageObserver
from common.argparsers import args_experiment
from common import transponder
from finetune import main as finetune_main
from compress import main as compress_main
from evaluate import main as evaluate_main
from analyze import main as analyze_main

TASK_FUNCS = {
    "finetune": finetune_main, "compress": compress_main,
    "evaluate": evaluate_main, "analyze": analyze_main
}

def run_experiment(task_args, _run):
    for task in task_args:
        TASK_FUNCS[task](task_args[task], sacred_experiment=_run)

def main(experiment_args, task_args):
    experiment_args, task_args = args_experiment()

    print(stdout.encoding)
    if stdout.encoding != "utf-8" and stdout.encoding != "UTF-8" :
        raise UnicodeError(f"Stdout encoding is {stdout.encoding} (should be utf-8)!")

    experiment = Experiment(experiment_args.name)

    output_dir = "experiments"

    if experiment_args.output_path is not None:
        output_dir = experiment_args.output_path

    os.makedirs(output_dir, exist_ok=True)

    output_storage = FileStorageObserver(output_dir)

    experiment.observers.append(output_storage)

    run_id = experiment_args.name
    if os.path.exists(f"{output_dir}/{run_id}"):
        if experiment_args.overwrite:
            rmtree(f"{output_dir}/{run_id}")
        else:
            previous_run = glob(f"{output_dir}/{run_id}*")
            index = previous_run[-1].split("_")[-1]
            try:
                index = int(index) + 1
            except ValueError:
                index = 2
            run_id = f"{run_id}_{index}"

            # Update model_name parameter for evaluate and analyze.
            for task_type in ("evaluate", "analyze"):
                if task_type in task_args and task_args[task_type].model_name is not None:
                    setattr(task_args[task_type], "model_name", run_id)

    experiment.add_config({
        "task_args": task_args
    })
    experiment.command(run_experiment)

    transponder.TRANSPONDER_ACTIVE = experiment_args.transponder

    transponder_args = None
    max_epochs = None

    if "finetune" in task_args:
        finetune_args = task_args["finetune"]
        max_epochs = finetune_args.max_epochs
        transponder_args = dict(finetune_args.__dict__)
        del transponder_args["max_epochs"]
    elif "compress" in task_args and "distill" in task_args["compress"].compression_actions:
        distillation_args = task_args["compress"]
        max_epochs = distillation_args.epochs
        transponder_args = dict(distillation_args.__dict__)
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
        transponder.send_train_start(run_id, transponder_args, max_epochs)

    run = experiment._create_run("run_experiment", info={"name": run_id})
    run._id = run_id
    print("test")
    exit()
    try:
        run()
    except UnicodeDecodeError:
        pass

if __name__ == "__main__":
    EXPERIMENT_ARGS, TASK_ARGS = args_experiment()
    main(EXPERIMENT_ARGS, TASK_ARGS)
