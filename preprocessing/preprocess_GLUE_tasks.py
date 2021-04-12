from sys import argv
import os
import io
import shutil
import requests
from common.task_utils import TASK_INFO

BPEMB_PATH = f"data/bpemb"

def download_bpemb_file(filename):
    os.makedirs(BPEMB_PATH, exist_ok=True)

    filepath = f"{BPEMB_PATH}/{filename}"
    response = requests.get(f"https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/{filename}")
    with open(filepath, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    print(f"Downloaded BPEmb file '{filename}'.", flush=True)

def format_mrpc(task_folder):
    mrpc_train_file = f"{task_folder}/msr_paraphrase_train.txt"
    mrpc_test_file = f"{task_folder}/msr_paraphrase_test.txt"

    with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
            io.open(os.path.join(task_folder, "test.tsv"), 'w', encoding='utf-8') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))

    dev_ids = []
    with io.open(os.path.join(task_folder, "mrpc_dev_ids.tsv"), encoding='utf-8') as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
         io.open(os.path.join(task_folder, "train.tsv"), 'w', encoding='utf-8') as train_fh, \
         io.open(os.path.join(task_folder, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

def preprocess_glue_task(glue_task):
    for bpemb_file in ("encoder.json", "vocab.bpe", "dict.txt"):
        if not os.path.exists(f"{BPEMB_PATH}/{bpemb_file}"):
            download_bpemb_file(bpemb_file)

    tasks = [glue_task]

    if glue_task == "ALL":
        tasks = TASK_INFO.keys()

    for task in tasks:
        print(f"Preprocessing '{task}'", flush=True)
        task_folder = TASK_INFO[task]["path"]
        if task == "mrpc":
            format_mrpc(task_folder)

        splits = ["train", "dev", "test"]
        input_count = 2

        if task == "qqp":
            input_columns = (3, 4)
            test_input_columns = (1, 2)
            label_column = 5
        elif task == "mnli":
            splits = ["train", "dev_matched", "dev_mismatched", "test_matched", "test_mismatched"]
            input_columns = (8, 9)
            test_input_columns = (8, 9)
            dev_label_column = 15
            label_column = 11
        elif task == "qnli":
            input_columns = (1, 2)
            test_input_columns = (1, 2)
            label_column = 3
        elif task == "mrpc":
            input_columns = (3, 4)
            test_input_columns = (3, 4)
            label_column = 0
        elif task == "rte":
            input_columns = (1, 2)
            test_input_columns = (1, 2)
            label_column = 3
        elif task == "sts-b":
            input_columns = (7, 8)
            test_input_columns = (7, 8)
            label_column = 9
        elif task == "sst-2":
            input_columns = (0,)
            test_input_columns = (1,)
            label_column = 1
            input_count = 1
        elif task == "cola":
            input_columns = (3,)
            test_input_columns = (1,)
            label_column = 1
            input_count = 1

        processed_folder = f"{task_folder}/processed"

        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)

        for split in splits:
            if task == "cola" and split != "test":
                shutil.copy(f"{task_folder}/{split}.tsv", f"{processed_folder}/{split}.tsv.temp")
            else:
                with open(f"{task_folder}/{split}.tsv", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.tsv.temp", "w", encoding="utf-8") as fp_out:
                        for index, line in enumerate(fp_in):
                            if index > 0:
                                fp_out.write(line.strip() + "\n")

            if task == "qqp" and split != "test":
                with open(f"{processed_folder}/{split}.tsv.temp", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.tsv", "w", encoding="utf-8") as fp_out:
                        for line in fp_in:
                            if len(line.strip().split("\t")) == 6:
                                fp_out.write(line.strip() + "\n")
            else:
                shutil.copy(f"{processed_folder}/{split}.tsv.temp", f"{processed_folder}/{split}.tsv")

            os.remove(f"{processed_folder}/{split}.tsv.temp")

        # Split into input0, input1 and label
        for split in splits:
            for input_type in range(input_count):
                if not split.startswith("test"):
                    column_number = input_columns[input_type]
                else:
                    column_number = test_input_columns[input_type]
                with open(f"{processed_folder}/{split}.tsv", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.raw.input{input_type}", "w", encoding="utf-8") as fp_out:
                        for line in fp_in:
                            fp_out.write(line.strip().split("\t")[column_number] + "\n")
            if not split.startswith("test"):
                with open(f"{processed_folder}/{split}.tsv", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.label", "w", encoding="utf-8") as fp_out:
                        col = dev_label_column if task == "mnli" and split != "train" else label_column
                        for line in fp_in:
                            fp_out.write(line.strip().split("\t")[col] + "\n")

            # BPE encode
            print(f"Running BPE encoding on '{task}' for '{split}' dataset", flush=True)
            for input_type in range(input_count):
                lang = f"input{input_type}"
                os.system(
                    "python -m examples.roberta.multiprocessing_bpe_encoder "
                    f"--encoder-json {BPEMB_PATH}/encoder.json --vocab-bpe {BPEMB_PATH}/vocab.bpe "
                    f"--inputs {processed_folder}/{split}.raw.{lang} "
                    f"--outputs {processed_folder}/{split}.{lang} "
                    "--workers 2 --keep-empty"
                )

        bin_path = f"{processed_folder}/{task}-bin"

        if os.path.exists(bin_path):
            shutil.rmtree(bin_path)

        devpref = f"{processed_folder}/dev.LANG"
        testpref = f"{processed_folder}/test.LANG"

        if task == "mnli":
            devpref = f"{processed_folder}/dev_matched.LANG,{processed_folder}/dev_mismatched.LANG"
            testpref = f"{processed_folder}/test_matched.LANG,{processed_folder}/test_mismatched.LANG"

        for input_type in range(input_count):
            lang = f"input{input_type}"
            os.system(
                f"fairseq-preprocess --only-source --trainpref {processed_folder}/train.{lang} "
                f"--validpref {devpref.replace('LANG', lang)} " +
                f"--testpref {testpref.replace('LANG', lang)} " +
                f"--destdir {bin_path}/{lang} --workers 2 --srcdict {BPEMB_PATH}/dict.txt"
            )

        if task != "sts-b":
            os.system(
                f"fairseq-preprocess --only-source --trainpref {processed_folder}/train.label "
                f"--validpref {devpref.replace('LANG', 'label')} " +
                f"--destdir {bin_path}/label --workers 2"
            )
        else:
            os.makedirs(f"{bin_path}/label")
            with open(f"{processed_folder}/train.label", "r", encoding="utf-8") as fp_in:
                with open(f"{bin_path}/label/train.label", "w", encoding="utf-8") as fp_out:
                    for line in fp_in:
                        fp_out.write(str(float(line.strip().split("\t")[0]) / 5.0) + "\n")
            with open(f"{processed_folder}/dev.label", "r", encoding="utf-8") as fp_in:
                with open(f"{bin_path}/label/valid.label", "w", encoding="utf-8") as fp_out:
                    for line in fp_in:
                        fp_out.write(str(float(line.strip().split("\t")[0]) / 5.0) + "\n")

        print(f"Done with '{task}'", flush=True)

if __name__ == "__main__":
    if len(argv) < 2:
        print("Run as following:")
        print("./examples/roberta/preprocess_GLUE_tasks.sh <task_name>")
        exit(1)

    preprocess_glue_task(argv[1])
