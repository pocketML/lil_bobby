from glob import glob
import json

import numpy as np

# Which models to save further compression results for.
EXTRA_COMPRESSION_MODELS = [
    [ # BiLSTM
        "bilstm_sst_alpha0_hash100_may25",
        "bilstm_qqp_alpha0_hash100_june1",
        "bilstm_mnli_alpha0_hash100_june9"
    ],
    [
        "bilstm_sst_alpha05_bpe100_may25",
        "bilstm_qqp_alpha05_bpe100_june1",
        "bilstm_mnli_alpha05_bpe100_june9"
    ],
    [
        "bilstm_sst_alpha0_bpe25_may27",
        "bilstm_qqp_alpha0_bpe25_june2",
        "bilstm_mnli_alpha0_bpe25_june9"
    ],
    [ # RNN
        "rnn_sst_alpha0_hash100_may27",
        "rnn_qqp_alpha0_hash100_june6",
        "rnn_mnli_alpha0_hash100_june9"
    ],
    [
        "rnn_sst_alpha0_bpe100_may27",
        "rnn_qqp_alpha0_bpe100_june6",
        "rnn_mnli_alpha0_bpe100_june9"
    ],
    [
        "rnn_sst_alpha0_char100_may28",
        "rnn_qqp_alpha0_char100_june8",
        "rnn_mnli_alpha0_char100_june10"
    ],
    [ # FFN
        "embffn_sst_alpha0_hash25_may28",
        "embffn_qqp_alpha0_hash25_june7",
        "embffn_mnli_alpha0_hash25_june10"
    ],
    [
        "embffn_sst_alpha1_bpe100_may29",
        "embffn_qqp_alpha1_bpe100_june7",
        "embffn_mnli_alpha1_bpe100_june10"
    ]
]

def get_experiment_suffix(result_name):
    try:
        suffix = int(result_name.split("_")[-1])
        return suffix
    except ValueError:
        return 1

def group_results_by_model(results_for_day):
    grouped_results = {}
    for result in results_for_day:
        split = result.split("_")
        if "og" in result:
            end = -2 if len(split) == 8 else -1
        else:
            end = -2 if len(split) == 7 else -1

        name = "_".join(split[:end])
        if name not in grouped_results or len(grouped_results[name]) == 4:
            grouped_results[name] = []
        grouped_results[name].append(result)
    return grouped_results

def get_extra_compression_results(table):
    compress_method = None
    if table == "final":
        compress_method = "prune_quant"
    elif table == "quantize":
        compress_method = "quant"

    models = []
    for model_group in EXTRA_COMPRESSION_MODELS:
        model_groups = []
        for task_specific_model in model_group:
            if table != "quantize":
                task_specific_model = "_".join(task_specific_model.split("_")[:-1])
            file_suffix = f"{compress_method}_*" if table != "prune" else "july*"
            results = glob(f"../experiments/{task_specific_model}_{file_suffix}")
            if len(results) < 4:
                continue
            results.sort(key=get_experiment_suffix)
            model_groups.append(results[-4:])
        models.append(model_groups)
    return models

def get_distillation_results():
    month_names = ["may", "june", "july", "august"]
    start_day_in_month = [25, 0, 0, 0]
    end_day_in_month = [31, 30, 31, 31]

    new_results = []
    for month_index, month in enumerate(month_names):
        for day in range(start_day_in_month[month_index], end_day_in_month[month_index] + 1):
            results_for_day = glob(f"../experiments/*_{month}{day}*")
            results_for_day.sort(key=get_experiment_suffix)

            grouped_results = group_results_by_model(results_for_day)

            for group in grouped_results:
                new_results.append(grouped_results[group])
    return new_results

def validate_experiment(data, table):
    compress_actions = [table]
    if table == "prune": # Get training aware pruning results.
        compress_actions = ["distill", "prune"]
    elif table == "final":
        compress_actions = ["prune", "quantize"]

    for comp_action in compress_actions:
        if comp_action not in data["compression_actions"]:
            return False

    expected_params = [
        ("embedding_freeze", [False, None]),
        ("embedding_dim", [25, 100, 300, None]),
        ("embedding_type", ["hash", "bpe", "char", None])
    ]
    if table == "distill":
        expected_params.append(("vocab_size", [5000, None]))

    for param, expected_values in expected_params:
        if data[param] not in expected_values:
            return False
    return True

def get_experiment_data(experiment_group, table):
    valid_groups = []
    for group_index in range(0, len(experiment_group) // 4, 4):
        with open(f"{experiment_group[group_index]}/config.json", "r") as fp:
            config = json.load(fp)["task_args"]["compress"]

        if validate_experiment(config, table):
            valid_groups.extend(experiment_group[group_index:group_index+4])

    if valid_groups == []:
        return None

    metrics = []
    if len(valid_groups) > 4:
        valid_groups = valid_groups[-4:]

    for experiment_path in valid_groups:
        with open(f"{experiment_path}/metrics.json", "r") as fp:
            metrics.append(json.load(fp))

    accuracies_1 = []
    accuracies_2 = []
    sizes = []
    theoretical_sizes = []
    try:
        for metrics_data in metrics:
            accuracy_1 = None
            accuracy_2 = None
            for key in ("test.accuracy", "test.matched.accuracy"):
                if key in metrics_data:
                    accuracy_1 = metrics_data[key]["values"][0]
            if accuracy_1 is None and "validation.acc" in metrics_data:
                accuracy_1 = max(metrics_data["validation.acc"]["values"])
            for key in ("test.f1", "test.mismatched.accuracy"):
                if key in metrics_data:
                    accuracy_2 = metrics_data[key]["values"][0]
            accuracies_1.append(accuracy_1)
            if accuracy_2 is not None:
                accuracies_2.append(accuracy_2)

            sizes.append(metrics_data["model_disk_size"]["values"][0])
            if "theoretical_size" in metrics_data:
                theoretical_sizes.append(metrics_data["theoretical_size"]["values"][0])
        params = metrics[0]["model_params"]["values"][0]
    except KeyError:
        return None

    mean_size = np.mean(np.array(sizes))
    mean_theoretical = None
    if theoretical_sizes != []:
        mean_theoretical = np.mean(np.array(theoretical_sizes))

    mean_1 = np.mean(np.array(accuracies_1))
    mean_2 = None if accuracies_2 == [] else np.mean(np.array(accuracies_2))

    std_1 = np.std(np.array(accuracies_1))
    std_2 = None if accuracies_2 == [] else np.std(np.array(accuracies_2))

    data_for_experiment = {
        "task": config["task"], "arch": config["student_arch"], "emb-type": config["embedding_type"],
        "emb-dim": config["embedding_dim"], "alpha": config.get("alpha"), "og": config.get("only_original_data"),
        "params": params, "size": mean_size, "theoretical_size": mean_theoretical,
        "acc": (mean_1, mean_2), "std": (std_1, std_2)
    }
    return data_for_experiment
