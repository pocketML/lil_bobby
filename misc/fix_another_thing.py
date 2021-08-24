import numpy as np

import evaluate

class SacredMock:
    def __init__(self):
        self.metrics = {}

    def log_scalar(self, key, scalar):
        self.metrics[key] = scalar

class ArgsMock:
    def __init__(self, task, model_name):
        self.task = task
        self.arch = "emb-ffn"
        self.cpu = False
        self.model_name = model_name
        self.loadbar = False

models = [
    "embffn_sst_alpha05_hash25_july12"
]

for task in ("sst", "qqp", "mnli"):
    results_for_task = []
    for alpha in ("1", "05", "0"):
        base_name = f"embffn_{task}_alpha{alpha}_hash25"
        suffixes = ["july12"]
        if alpha == "1":
            suffixes = ["og_july12"] + suffixes

        accuracies_1 = []
        accuracies_2 = []
        for suffix in suffixes:    
            for seed_name in ("bennington", "hadfield", "feynman", "simone"):
                full_name = f"{base_name}_{suffix}"
                args_mock = ArgsMock(task, full_name)
                sacred_mock = SacredMock()
                evaluate.main(args_mock, sacred_experiment=sacred_mock)

                if task in ("sst", "qqp"):
                    accuracies_1.append(sacred_mock.metrics["test.accuracy"])
                if task == "qqp":
                    accuracies_2.append(sacred_mock.metrics["test.f1"])
                elif task == "mnli":
                    accuracies_1.append(sacred_mock.metrics["test.matched.accuracy"])
                    accuracies_2.append(sacred_mock.metrics["test.mismatched.accuracy"])

        mean_1 = np.mean(accuracies_1) * 100
        std_dev_1 = np.std(accuracies_1) * 100
        if accuracies_2 != []:
            mean_2 = np.mean(accuracies_2) * 100
            std_dev_2 = np.std(accuracies_2) * 100

        acc_text = f"{mean_1:.1f}"
        if accuracies_1 != []:
            acc_text += f"/{mean_2:.1f}"

        sd_text = f"{std_dev_1:.1f}"
        if accuracies_1 != []:
            sd_text += f"/{std_dev_2:.1f}"

        results_for_task.append(acc_text)
        results_for_task.append(sd_text)

    print(" & ".join(results_for_task))
