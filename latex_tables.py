import argparse
import json
from glob import glob

import numpy as np
from numpy.core.fromnumeric import std

OLD_RESULT_NAMES = [
    # BPE 25
    "tang_sst_bpe25_may14", "tang_bpe_dim25_alpha1_may9",
    "tang_sst_alpha05_bpe25_may16", "tang_sst_alpha0_bpe25_may16",
    # BPE 300
    "tang_ogdata_alpha1_may12", "tang_bpe_alpha1_may6",
    "tang_sst_alpha05_bpe300_may16", "tang_sst_alpha0_bpe300_may16",
    # HASH 25
    "tang_sst_hash25_og_may14", "tang_sst_hash25_may14",
    "tang_sst_alpha05_hash25_may17", "tang_sst_alpha0_hash25_may17",
    # HASH 300
    "tang_sst_hash300_og_may14", "tang_sst_hash300_may14",
    "tang_sst_alpha05_hash300_may17", "tang_sst_alpha0_hash300_may17"
]

def get_old_results():
    old_results = []
    
    for name in OLD_RESULT_NAMES:
        path = f"experiments/{name}"
        old_results.append(
            [f"{path}_bennington", f"{path}_hadfield", f"{path}_feynman", f"{path}_simone"]
        )
    return old_results

def get_new_results():
    month_names = ["may", "june", "july", "august"]
    start_day_in_month = [25, 0, 0, 0]
    end_day_in_month = [31, 30, 31, 31]

    new_results = []
    for month_index, month in enumerate(month_names):
        for day in range(start_day_in_month[month_index], end_day_in_month[month_index]):
            results_for_day = glob(f"experiments/*_{month}{day}")
            grouped_results = {}
            for result in results_for_day:
                split = result.split("_")
                name = "_".join(split[:-1])
                if name not in grouped_results:
                    grouped_results[name] = []
                grouped_results[name].append(result)

            for group in grouped_results:
                new_results.append(grouped_results[group])
    return new_results

def get_experiment_data(experiment_group):
    metrics = []
    with open(f"{experiment_group[0]}/config.json", "r") as fp:
        config = json.load(fp)

    for experiment_path in experiment_group:
        with open(f"{experiment_path}/metrics.json", "r") as fp:
            metrics.append(json.load(fp))

    accuracies_1 = []
    accuracies_2 = []

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
        accuracies_2.append(accuracy_2)

    params = metrics_data[0]["model_params"]["values"][0]
    disk_size = metrics_data[0]["model_disk_size"]["values"][0]

    mean_1 = np.mean(np.array(accuracies_1))
    mean_2 = None if accuracies_2 == [] else np.mean(np.array(accuracies_2))

    std_1 = np.std(np.array(accuracies_1))
    std_2 = None if accuracies_2 == [] else np.std(np.array(accuracies_2))

    data_for_experiment = {
        "arch": config["student_arch"], "emb-type": config["embedding_type"],
        "emb-dim": config["embedding_dim"], "alpha": config["alpha"], "og": config["original_data"],
        "params": params, "size": disk_size, "acc": (mean_1, mean_2), "std": (std_1, std_2)
    }
    return data_for_experiment

def group_and_format_data(results):
    alpha_indices = {0: 1, 0.5: 2, 1: 3}
    emb_sort_order = [
        "hash", "bpe", "char"
    ]

    grouped_data = {}
    for result_group in results:
        data = get_experiment_data(result_group)
        grouped_data[data["task"]] = data

    for task in grouped_data:
        grouped_by_arch = {
            "bilstm": [], "rnn": [], "emb-ffn": []
        }
        for data in grouped_data[task]:
            grouped_by_arch[data["arch"]] = data
        grouped_data[task] = grouped_by_arch

        for arch in grouped_data[task]:
            grouped_by_emb = {}
            for data in grouped_data[task][arch]:
                key = f"{data['emb-type']}_{data['emb-dim']}"

                if key not in grouped_by_emb:
                    grouped_by_emb[key] = {}
                    grouped_by_emb[key]["acc"] = ["" for _ in range(4)]
                    grouped_by_emb[key]["std"] = ["" for _ in range(4)]

                alpha_index = 0 if data["og"] else alpha_indices[data["alpha"]]

                grouped_by_emb[key][alpha_index] = {
                    data['emb-type'], data['emb-dim'],
                    data["params"], data["size"],
                    data["acc"], data["std"]
                }

                grouped_by_emb[key]["acc"][alpha_index] = data["acc"]
                grouped_by_emb[key]["std"][alpha_index] = data["std"]

            for key in grouped_by_emb:
                interleaved_measurements = [0] * 8
                for measurement in ("acc", "std"):
                    max_measure_index_1 = 0
                    max_measure_1 = 0
                    max_measure_index_2 = 0
                    max_measure_2 = 0
                    for index in range(grouped_by_emb[key][measurement]):
                        val_1, val_2 = grouped_by_emb[key][measurement][index]
                        if val_1 > max_measure_1:
                            max_measure_1 = val_1
                            max_measure_index_1 = index
                        if val_2 is not None and val_2 > max_measure_2:
                            max_measure_2 = val_2
                            max_measure_index_2 = index

                    for index in range(grouped_by_emb[key][measurement]):
                        val_1, val_2 = grouped_by_emb[key][measurement][index]
                        fmt_val = f"{val_1:.1f}"
                        if index == max_measure_index_1:
                            fmt_val = "\\textbf{" + fmt_val + "}"
                        if val_2 is not None:
                            fmt_val_2 = f"{val_2:.1f}"
                            if index == max_measure_index_2:
                                fmt_val_2 = "\\textbf{" + fmt_val_2 + "}"
                            fmt_val = fmt_val + f"/{fmt_val_2}"

                        interleaved_index = index * 2
                        if measurement == "std":
                            interleaved_index += 1
                        interleaved_measurements[interleaved_index] = fmt_val
                    
                    grouped_by_emb[key]["measurements"] = interleaved_measurements

            emb_group_list = list(grouped_by_emb.values())
            emb_group_list.sort(key=lambda x: (emb_sort_order.index(x["emb-type"], x["emb-dim"])))

            grouped_data[task][arch] = emb_group_list

    return grouped_data

def print_table(grouped_data):
    arch_formatted = {
        "bilstm": "BiLSTM",
        "rnn": "RNN",
        "emb-ffn": "FFN"
    }
    emb_formatted = {
        "hash": "Hash",
        "bpe": "BPE",
        "char": "Char"
    }

    # Format and size things.
    print("{")
    print("\\centering")
    print("\\begin{table*}[!htb]")
    print("\\centering")
    print("\\setlength\\tabcolsep{4pt}")
    print("\\begin{footnotesize}")
    print("\\renewcommand{\\arraystretch}{1.3}")

    # Start of table
    print("\\begin{tabular}{c||c|c|c|c||cc|cc|cc|cc}")
    print("\\hline")

    # Print headers
    header_line = (
        "\\multirow{2}{*}{Type} & \\multirow{2}{*}{\\textit{E}} & " +
        "\\multirow{2}{*}{\\textit{D}} & \\multirow{2}{*}{\\textit{P}} & " + 
        "\\multirow{2}{*}{\\textit{S}} & \\multicolumn{2}{c|}{No distill} & " +
        "\\multicolumn{2}{c|}{No distill + \\textit{b}} & " +
        "\\multicolumn{2}{c|}{Alpha 0.5 + \\textit{b}} & " + 
        "\\multicolumn{2}{c}{Alpha 0 + \\textit{b}} \\\\\n" +
        " &  &  &  &  & mean & sd & mean & sd & mean & sd & mean & sd \\\\"
    )
    print(header_line)
    print("\\hhline{=#=|=|=|=#==|==|==|==}")

    # Actually print the data
    for arch in grouped_data:
        line = "\multirow{2}{*}{" + arch_formatted[arch] + "}"
        for data in grouped_data[arch]:
            row_data = [
                emb_formatted[data["emb-type"]], data["emb-dim"],
                data["params"], data["size"]
            ]
            row_data = row_data + data["measurements"]

            line += "& ".join(row_data)
        print(line)

def main(args):
    new_results = get_new_results()
    old_results = get_old_results()

    all_results = old_results + new_results

    grouped_data = group_and_format_data(all_results)

    print_table(grouped_data[args.task])

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("--task", type=str, choices=("sst-2", "qqp", "mnli"))

    ARGS = PARSER.parse_args()

    main(ARGS)