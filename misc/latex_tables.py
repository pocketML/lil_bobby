import argparse
import load_results

import numpy as np

ARCH_FORMATTED = {
    "bilstm": "BiLSTM",
    "rnn": "RNN",
    "emb-ffn": "FFN"
}
EMB_FORMATTED = {
    "hash": "Hash",
    "bpe": "BPE",
    "char": "Char"
}

SIZE_ROBERTA = 1426.02 # MB
SIZE_GLUE = 170 # MB

def group_and_format_distill_data(results, table):
    alpha_indices = {1.0: 1, 0.5: 2, 0.0: 3}
    emb_sort_order = [
        "hash", "bpe", "char"
    ]

    grouped_data = {"sst-2": [], "qqp": [], "mnli": []}
    for result_group in results:
        data = load_results.get_experiment_data(result_group, table)
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
                vocabs = [5000]
                if emb_type == "hash" and emb_dim == 25 and arch == "emb-ffn":
                    vocabs.append(2500)

                for vocab in vocabs:
                    grouped_by_emb[f"{key}_{vocab}"] = {
                        "emb-type": emb_type,
                        "emb-dim": emb_dim,
                        "vocab-size": vocab,
                        "params": "", "size": "",
                        "acc": [(None, None) for _ in range(4)],
                        "std": [(None, None) for _ in range(4)]
                    }

            for data in grouped_data[task][arch]:
                key = f"{data['emb-type']}_{data['emb-dim']}_{data['vocab-size']}"

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
            emb_group_list.sort(
                key=lambda x: (-x["vocab-size"], emb_sort_order.index(x["emb-type"]), x["emb-dim"])
            )

            grouped_data[task][arch] = emb_group_list

    return grouped_data

def get_og_results(result_group, task):
    task_ids = {"sst-2": 0, "qqp": 1, "mnli": 2}
    for model_group in load_results.EXTRA_COMPRESSION_MODELS:
        task_index = task_ids[task]
        model_name = "_".join(model_group[task_index].split("_"))
        if model_name in result_group[0]:
            return [
                f"experiments/{model_group[task_index]}_{seed_name}"
                for seed_name in ("bennington", "hadfield", "feynman", "simone")
            ]
    return None

def format_compr_ratio(ratio):
    ratio_str = str(ratio)
    if len(ratio_str) < 4:
        return ratio_str + "x"

    str_with_spaces = ""
    for i in range(len(ratio_str), 0, -1):
        reverse_i = len(ratio_str) - i
        if reverse_i > 0 and i % 3 == 0:
            str_with_spaces += " "
        str_with_spaces += ratio_str[reverse_i]

    return str_with_spaces + "x"

def format_extra_compression_row(data, task):
    acc_1, acc_2 = data["acc"]
    acc_str = f"{(acc_1 * 100):.2f}"
    if acc_2 is not None:
        acc_str += f" / {(acc_2 * 100):.2f}"

    compr_ratio = None
    size_value = data['theoretical_size']
    if task in ("sst-2", "qqp"):
        compr_ratio = format_compr_ratio(int(SIZE_ROBERTA / size_value))

    return acc_str, f"{size_value:.3f}", compr_ratio

def group_and_format_extra_compression_data(results, table):
    all_model_data = []
    og_model_data = []
    for model_group in results:
        acc_data = []
        acc_data_og = []
        size_data = []
        size_data_og = []
        compr_ratio = []
        compr_ratio_og = []

        tasks = ["sst-2", "qqp", "mnli"]
        for task, result_group in zip(tasks, model_group):
            compress_data = load_results.get_experiment_data(result_group, table)
            og_results = get_og_results(result_group, task)
            og_data = load_results.get_experiment_data(og_results, "distill")

            if task == "sst-2": # Only add arch once.
                emb_fmt = EMB_FORMATTED[og_data["emb-type"]]
                model_id = f"{emb_fmt}{og_data['emb-dim']}"
                arch_fmt = ARCH_FORMATTED[og_data["arch"]]
                model_name = f"{arch_fmt}\\textsubscript" + "{" + model_id + "}"
                compress_denote = table[0] if table != "final" else "p + q"
                acc_data.append(f"{model_name} + {compress_denote}")
                acc_data_og.append(model_name)

            acc, size, ratio = format_extra_compression_row(compress_data, task)
            acc_og, size_og, ratio_og = format_extra_compression_row(og_data, task)
 
            acc_data.append(acc)
            acc_data_og.append(acc_og)

            if task in ("sst-2", "qqp"):
                size_data.append(size)
                size_data_og.append(size_og)
                compr_ratio.append(ratio)
                compr_ratio_og.append(ratio_og)

        all_model_data.append(acc_data + size_data + compr_ratio)
        og_model_data.append(acc_data_og + size_data_og + compr_ratio_og)

    return all_model_data, og_model_data

def print_extra_compression_table(grouped_data, table):
    print("{")
    print("\\centering")
    print("\\begin{table*}[!htb]")
    print("\\centering")
    print("\\begin{footnotesize}")

    # Start of table
    print("\\begin{tabular}{l||c|c|c||rr|rr}")
    print("\\hline")

    # Print headers
    header_line = (
        "Model & SST-2 & QQP & MNLI & \\multicolumn{2}{c}{Size (SS/SP)} & "
        "\\multicolumn{2}{c}{Compr. rate (SS/SP)}\\\\"
    )
    print(header_line)
    print("\\hline")

    roberta_data = [
        "RoBERTa\\textsubscript{Large}", "96.56", "92.15 / 89.58", "90.33", "1426.02", "1426.02", "1x & 1x"
    ]

    # Actually print the data
    print(" & ".join(roberta_data) + "\\\\")

    if table == "final":
        glue_data_base = [
            "GLUE Baseline + GLoVE", "87.5", "85.3 / 82.0", "66.7", "2946.73", "2946.7", "2x & 2x"
        ]
        print(" & ".join(glue_data_base) + "\\\\")

        glue_data_elmo = [
            "GLUE Baseline + Elmo", "91.5", "88.0 / 84.3", "68.6", "681.13", "681.13", "2x & 2x"
        ]
        print(" & ".join(glue_data_elmo) + "\\\\")

        tinybert_data = [
            "TinyBERT\\textsubscript{4}", "93.0", "88.00 / 91.10", "84.50", "268.00", "268.00", "5x & 5x"
        ]
        print(" & ".join(tinybert_data) + "\\\\")

    print("\\hline")

    # Print original data
    for model_data_og, model_data_prune, model_data_quant, model_data_final in zip(*grouped_data):
        print(" & ".join(model_data_og) + "\\\\")
        print(" & ".join(model_data_prune) + "\\\\")
        print(" & ".join(model_data_quant) + "\\\\")
        print(" & ".join(model_data_final) + "\\\\")
        print("\\hline")

    print("\\end{tabular}")

    if table in ("prune", "final"):
        compress_desc = "quantized/pruned" if table == "final" else "pruned"
        task_desc = (
            f"Size of original models is disk size in MB. Size of {compress_desc} models " +
            "is disk size in MB of zipped model using gzip deflate algorithm."
        )

    # Caption stuff.
    if table == "prune":
        task_name = "pruning"
    elif table == "quantize":
        task_name = "quantization"
        task_desc = "Size is disk size of model in MB"
    elif table == "final":
        task_name = "pruning \& quantization"

    caption = (
        f"Results for {task_name} of selected distilled models. Performances are measured by " +
        "accuracy, accuracy/f1-score, matched/mismatched accuracy for the three tasks, " +
        f"respectively. {task_desc}. Compression ratio is single sentence/sentence pair models " +
        "compared to the disk size of teacher model RoBERTa Large. " +
        "\\textit{SS}: Single Sentence. \\textit{SP}: Sentence Pair."
    )
    print("\\caption{" + caption + "}")
    print("\\label{tab:" + task_name.replace(" \& ", "_") + "_results}")
    print("\\end{footnotesize}")
    print("\\end{table*}")
    print("}")

def print_distill_table(grouped_data, task):
    # Format and size things.
    print("{")
    print("\\centering")
    print("\\begin{table*}[!htb]")
    print("\\centering")
    if task == "qqp":
        print("\\setlength{\\tabcolsep}{3pt}")
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
        line += "{" + ARCH_FORMATTED[arch] + "} & "
        for index, data in enumerate(grouped_data[task][arch]):
            emb_name = EMB_FORMATTED[data["emb-type"]]
            if data["vocab-size"] == 2500:
                emb_name = emb_name + "*"

            row_data = [
                emb_name, str(data["emb-dim"]), data["params"], data["size"]
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
        task_specific_caption = (
            "Performances written as average of \\textbf{matched accuracy/mismatched accuracy}. "
        )

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
        "for that combination of embedding type and dimension. *Vocab Size of 2500.}"
    )
    print(caption_text)
    short_task = "sst" if task == "sst-2" else task
    print("\\label{tab:distillation_results_" + short_task + "}")
    print("\\end{table*}")
    print("}")

def main(args):
    if args.table == "distill":
        all_results = load_results.get_distillation_results()
        grouped_data = group_and_format_distill_data(all_results, args.table)
        print_distill_table(grouped_data, args.task)
    else:
        all_table_data = []
        for table in ("prune", "quantize", "final"):
            all_results = load_results.get_extra_compression_results(table)
            compress_data, og_data = group_and_format_extra_compression_data(all_results, table)
            if table == "prune":
                all_table_data.append(og_data)
            all_table_data.append(compress_data)
        print_extra_compression_table(all_table_data, args.table)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument("task", type=str, choices=("sst-2", "qqp", "mnli"))
    PARSER.add_argument("table", type=str, choices=("distill", "prune", "quantize", "final"))

    ARGS = PARSER.parse_args()

    main(ARGS)
