import os
from sys import stdout
from shutil import rmtree
from argparse import ArgumentError

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver

from common import transponder
from common.argparsers import args_experiment
from finetune import main as finetune_main
from compress import main as compress_main
from evaluate import main as evaluate_main
from analyze import main as analyze_main

TASK_FUNCS = { # Tasks that can be run in the cotext of a Sacred experiment.
    "finetune": finetune_main, "compress": compress_main,
    "evaluate": evaluate_main, "analyze": analyze_main
}

def run_experiment(task_args, _run):
    """
    Run the given tasks in the context of the given Sacred experiment.

    Parameters
    ----------
    task_args : dict[str, Namespace]
        Dictionary mapping task names ('evaluate', 'compress', etc.)
        to arguments for that task.
    _run : Run
        Sacred experiment instance to save results to for the given tasks.
    """
    metrics_filename = f"experiments/{_run.info['name']}/metrics.json"
    if not os.path.exists(metrics_filename):
        # Sometimes metrics.json is not created by Sacred... So we do it ourselves.
        with open(metrics_filename, "w", encoding="utf-8") as fp:
            fp.write("{}")

    # Run through task_args and execute the main function for each task.
    augmented_model = None
    for task in task_args:
        # Each main function for each task returns the model it operated on.
        augmented_model = TASK_FUNCS[task](
            task_args[task], sacred_experiment=_run, model=augmented_model
        )

def main(experiment_args, task_args):
    """
    Create and run a Sacred experiment.
    
    Parameters
    ----------
    experiment_args : Namespace
        Arguments for running the experiment.
        Includes what tasks to run, name of experiments, etc.
    task_args : dict[str, Namespace]
        Dictionary mapping task names ('evaluate', 'compress', etc.)
        to arguments for that task.
    """
    SETTINGS["CAPTURE_MODE"] = "no" # Disable stdout capturing by Sacred as it causes issues.

    if stdout.encoding != "utf-8" and stdout.encoding != "UTF-8" : # Ensure stdout utf-8 encoding.
        raise UnicodeError(f"Stdout encoding is {stdout.encoding} (should be utf-8)!")

    # Create Sacred experiment and associated experiment folders.
    experiment = Experiment(experiment_args.name)

    output_dir = "experiments"

    if experiment_args.output_path is not None:
        output_dir = experiment_args.output_path

    os.makedirs(output_dir, exist_ok=True)

    # Create Sacred file storage observer
    output_storage = FileStorageObserver(output_dir)

    experiment.observers.append(output_storage)

    # Check if an experiment with same name exists.
    run_id = experiment_args.name
    if os.path.exists(f"{output_dir}/{run_id}"):
        if experiment_args.overwrite:
            # Remove old experiment if 'overwrite' is true.
            rmtree(f"{output_dir}/{run_id}")
        else:
            index = 2
            # Append unique index to experiment name.
            while os.path.exists(f"{output_dir}/{run_id}_{index}"):
                index += 1
            run_id = f"{run_id}_{index}"

            if "compress" in task_args or "finetune" in task_args:
                # Update model_name parameter with new experiment name for evaluate and analyze.
                for task_type in ("evaluate", "analyze"):
                    if task_type in task_args and task_args[task_type].model_name is not None:
                        setattr(task_args[task_type], "model_name", run_id)

    # Create config for Sacred. This sets variables that can be accessed later.
    sacred_cfg = {"task_args": task_args}
    if "compress" in task_args:
        sacred_cfg["seed"] = task_args["compress"].seed
    elif "finetune" in task_args:
        sacred_cfg["seed"] = task_args["finetune"].seed

    experiment.add_config(sacred_cfg)
    experiment.command(run_experiment)

    # Activate transponder if it should be used. This sends training updates to a webserver.
    transponder.TRANSPONDER_ACTIVE = experiment_args.transponder

    transponder_args = None
    max_epochs = None

    # Filter training arguments and send relevant ones to webserver (if transponder is active).
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

    # Create experiment and run it.
    run = experiment._create_run("run_experiment", info={"name": run_id})
    run._id = run_id
    try:
        run()
    except UnicodeDecodeError:
        print("Unicode error for some reason.")

if __name__ == "__main__":
    EXPERIMENT_ARGS, TASK_ARGS, REMAIN = args_experiment()
    if len(REMAIN) > 0:
        raise ArgumentError(None, f"Couldn't parse the following arguments: {REMAIN}")
    main(EXPERIMENT_ARGS, TASK_ARGS)
