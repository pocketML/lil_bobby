import argparse
import compression
import json
from glob import glob

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
    compress_method = table
    if table == "final":
        compress_method = "prune_quant"
    elif table == "quantize":
        compress_method = "quant"

    models = []
    for model_group in EXTRA_COMPRESSION_MODELS:
        model_groups = []
        for task_specific_model in model_group:
            results = glob(f"experiments/{task_specific_model}_{compress_method}_*")
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
            results_for_day = glob(f"experiments/*_{month}{day}*")
            results_for_day.sort(key=get_experiment_suffix)

            grouped_results = group_results_by_model(results_for_day)

            for group in grouped_results:
                new_results.append(grouped_results[group])
    return new_results

def validate_experiment(data, table):
    compress_actions = ["prune", "quantize"] if table == "final" else [table]
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

    try:
        params = metrics[0]["model_params"]["values"][0]
        disk_size = metrics[0]["model_disk_size"]["values"][0]
        theoretical_size = metrics[0]["theoretical_size"]["values"][0]
    except KeyError:
        return None

    mean_1 = np.mean(np.array(accuracies_1))
    mean_2 = None if accuracies_2 == [] else np.mean(np.array(accuracies_2))

    std_1 = np.std(np.array(accuracies_1))
    std_2 = None if accuracies_2 == [] else np.std(np.array(accuracies_2))

    data_for_experiment = {
        "task": config["task"], "arch": config["student_arch"], "emb-type": config["embedding_type"],
        "emb-dim": config["embedding_dim"], "alpha": config.get("alpha"), "og": config.get("only_original_data"),
        "params": params, "size": disk_size, "theoretical_size": theoretical_size,
        "acc": (mean_1, mean_2), "std": (std_1, std_2)
    }
    return data_for_experiment

def group_and_format_distill_data(results, table):
    alpha_indices = {1.0: 1, 0.5: 2, 0.0: 3}
    emb_sort_order = [
        "hash", "bpe", "char"
    ]

    grouped_data = {"sst-2": [], "qqp": [], "mnli": []}
    for result_group in results:
        data = get_experiment_data(result_group, table)
        if data is not None:
            grouped_data[data["task"]].append(data)

    for task in grouped_data:
        grouped_by_arch = {
            "bilstm": [], "rnn": [], "emb-ffn": []
        }
        for data in grouped_data[task]:
            grouped_by_arch[data["arch"]].append(data)
        grouped_data[task] = grouped_by_arch

        for arch in grouped_data[task]:
            embeddings = [
                ("hash", 25), ("hash", 100), ("hash", 300),
                ("bpe", 25), ("bpe", 100), ("bpe", 300),
                ("char", 100)
            ]
            if task != "sst-2":
                embeddings.remove(("hash", 300))
                embeddings.remove(("bpe", 300))

            grouped_by_emb = {}

            for emb_type, emb_dim in embeddings:
                key = f"{emb_type}_{emb_dim}"
                grouped_by_emb[key] = {
                    "emb-type": emb_type,
                    "emb-dim": emb_dim,
                    "params": "", "size": "",
                    "acc": [(None, None) for _ in range(4)],
                    "std": [(None, None) for _ in range(4)]
                }

            for data in grouped_data[task][arch]:
                key = f"{data['emb-type']}_{data['emb-dim']}"

                fmt_params = np.format_float_scientific(data["params"], precision=1, exp_digits=1, trim="0")
                fmt_size = f"{data['size']:.2f}"
                grouped_by_emb[key]["params"] = fmt_params
                grouped_by_emb[key]["size"] = fmt_size

                alpha_index = 0 if data["og"] else alpha_indices[data["alpha"]]

                grouped_by_emb[key]["acc"][alpha_index] = data["acc"]
                grouped_by_emb[key]["std"][alpha_index] = data["std"]

            for key in grouped_by_emb:
                interleaved_measurements = [0] * 8
                for measurement in ("acc", "std"):
                    max_measure_index_1 = []
                    max_measure_1 = 0
                    max_measure_index_2 = []
                    max_measure_2 = 0
                    for index in range(len(grouped_by_emb[key][measurement])):
                        val_1, val_2 = grouped_by_emb[key][measurement][index]
                        if val_1 is not None and val_1 >= max_measure_1:
                            if val_1 > max_measure_1:
                                max_measure_index_1 = [index]
                            else:
                                max_measure_index_1.append(index)
                            max_measure_1 = val_1
                        if val_2 is not None and val_2 >= max_measure_2:
                            if val_1 > max_measure_2:
                                max_measure_index_2 = [index]
                            else:
                                max_measure_index_2.append(index)
                            max_measure_2 = val_2

                    for index in range(len(grouped_by_emb[key][measurement])):
                        val_1, val_2 = grouped_by_emb[key][measurement][index]
                        fmt_val = ""
                        if val_1 is not None:
                            fmt_val = f"{(val_1 * 100):.1f}"
                            if measurement == "acc" and index in max_measure_index_1:
                                fmt_val = "\\textbf{" + fmt_val + "}"
                        if val_2 is not None:
                            fmt_val_2 = f"{(val_2 * 100):.1f}"
                            if measurement == "acc" and index in max_measure_index_2:
                                fmt_val_2 = "\\textbf{" + fmt_val_2 + "}"
                            fmt_val = fmt_val + f"/{fmt_val_2}"

                        interleaved_index = index * 2
                        if measurement == "std":
                            interleaved_index += 1
                        interleaved_measurements[interleaved_index] = fmt_val

                    grouped_by_emb[key]["measurements"] = interleaved_measurements

            emb_group_list = list(grouped_by_emb.values())
            emb_group_list.sort(key=lambda x: (emb_sort_order.index(x["emb-type"]), x["emb-dim"]))

            grouped_data[task][arch] = emb_group_list

    return grouped_data

def group_and_format_extra_compression_data(results, table):
    all_model_data = []
    model_ids = ["a", "b", "c"]
    times_seen_arch = {"bilstm": 0, "rnn": 0, "embffn": 0}
    for model_group in results:
        acc_data = []
        disk_sizes = []
        theoretical_sizes = []

        tasks = ["sst-2", "qqp", "mnli"]
        for task, result_group in zip(tasks, model_group):
            data = get_experiment_data(result_group, table)
            if task == "sst-2": # Only add arch once.
                model_id_index = times_seen_arch[data["arch"]]
                times_seen_arch[data["arch"]] += 1
                model_id = model_ids[model_id_index]
                acc_data.append(f"{data['arch']}_{model_id}")
            acc_1, acc_2 = data["acc"]
            acc_str = str(acc_1)
            if acc_2 is not None:
                acc_str += f" / {acc_2}"

            acc_data.append(acc_str)

            if task in ("sst-2", "qqp"):
                disk_sizes.append(f"{data['size']:.2f}")
                if table in ("prune", "final"):
                    theoretical_sizes.append(f"{data['theoretical_size']:.2f}")
        
        data_for_model = acc_data + disk_sizes
        if table in ("prune", "final"):
            data_for_model.extend(theoretical_sizes)

        all_model_data.append(data_for_model)
    return all_model_data

def print_prune_table(grouped_data, task):
    """
    Columns:
        Model Name (explained in text after), SST-2 Acc, QQP Acc, MNLI Acc,
        Size on Disk (before/after pruning (zipped after)),
        Compression Ratio (before/after)
    Rows:
        The different models (described in Docs), pruned in some way.
    """

def print_distill_table(grouped_data, task):
    """
    Columns:
        Embedding Type, Embedding Dim, Parameters, Size,
        No Distill, No Distill + b, Alpha 0.5, Alpha 0
    Rows:
        The different combinations of distilled models with
        different embeddings/architectures
    """
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
    if task != "sst-2":
        print("\\setlength{\\tabcolsep}{4pt}")
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
    for arch in grouped_data[task]:
        line = "\\multirow{" + f"{len(grouped_data[task][arch])}" + "}{*}"
        line += "{" + arch_formatted[arch] + "} & "
        for index, data in enumerate(grouped_data[task][arch]):
            row_data = [
                emb_formatted[data["emb-type"]], str(data["emb-dim"]),
                data["params"], data["size"]
            ]
            row_data = row_data + data["measurements"]

            line += " & ".join(row_data) + "\\\\"
            if index < len(grouped_data[task][arch]) - 1:
                line += "\n & "

        print(line)
        print("\\hline")

    print("\\end{tabular}")
    print("\\renewcommand{\\arraystretch}{1}")
    print("\\end{footnotesize}")

    if task == "sst-2":
        task_specific_caption = "Performances written as \\textbf{accuracy}. "
    if task == "qqp":
        task_specific_caption = "Performances written as \\textbf{accuracy/f1-score}. "
    elif task == "mnli":
        task_specific_caption = "Performances written as \\textbf{matched accuracy/mismatched accuracy}. "

    # Table caption
    caption_text = (
        "\\caption{Results for models trained on " + task.upper() + " dataset. " +
        task_specific_caption +
        "Performances reported in \\textit{mean} and \\textit{sd} " +
        "(standard deviation) are measured in percentage and are from four " +
        "runs with different seeds. \\textit{E}: Embedding type. \\textit{D}: " +
        "Dimension of embedding vectors. \\textit{P}: Parameter count for the " +
        "entire model. \\textit{S}: Size on disk, in Megabytes. \\textit{b}: " +
        "Bootstrapped dataset. Bold: best mean across the four training methods " +
        "for that combination of embedding type and dimension.}"
    )
    print(caption_text)
    print("\\end{table*}")
    print("}")

def print_extra_compression_table(grouped_data):
    print(grouped_data[0])

def main(args):
    if args.table == "distill":
        all_results = get_distillation_results()
        grouped_data = group_and_format_distill_data(all_results, args.table)
        print_distill_table(grouped_data, args.task)
    else:
        all_results = get_extra_compression_results(args.table)
        grouped_data = group_and_format_extra_compression_data(all_results, args.table)
        print_extra_compression_table(grouped_data)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("task", type=str, choices=("sst-2", "qqp", "mnli"))
    PARSER.add_argument("table", type=str, choices=("distill", "prune", "quantize", "final"))

    ARGS = PARSER.parse_args()

    main(ARGS)
