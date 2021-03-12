from sacred import Experiment
from sacred.observers import FileStorageObserver

def run_experiment(experiment, name):
    print(f"Running {name}")
    experiment.log_scalar("val.accuracy", 10)
    experiment.add_artifact("checkpoints/finetuning_pruning_3/checkpoint1.pt")

NAME = "test123"

EXPERIMENT = Experiment(NAME)

OUTPUT_STORAGE = FileStorageObserver("experiments")

EXPERIMENT.observers.append(OUTPUT_STORAGE)

EXPERIMENT.add_config({
    "experiment": EXPERIMENT, "name": NAME
})
EXPERIMENT.command(run_experiment)

RUN = EXPERIMENT._create_run("run_experiment")
RUN._id = NAME
RUN()
