from sys import argv
import os
import shutil
import requests
from common.task_utils import TASK_INFO

BPEMB_PATH = f"data/bpemb"

def download_bpemb_file(filename):
    if not os.path.exists(BPEMB_PATH):
        os.mkdir(BPEMB_PATH)

    filepath = f"{BPEMB_PATH}/{filename}"
    response = requests.get(f"https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/{filename}")
    with open(filepath, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    print(f"Downloaded '{filename}'.")

def preprocess_glue_task(glue_task):
    for bpemb_file in ("encoder.json", "vocab.bpe", "dict.txt"):
        if not os.path.exists(f"{BPEMB_PATH}/{bpemb_file}"):
            download_bpemb_file(bpemb_file)

    tasks = [glue_task.replace("glue_", "")]

    if glue_task == "ALL":
        tasks = TASK_INFO.keys()

    for task in tasks:
        print(f"Preprocessing '{task}'")
        task_folder = TASK_INFO[task]["path"]

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
            input_count = 0
        elif task == "cola":
            input_columns = (3,)
            test_input_columns = (1,)
            label_column = 1
            input_count = 0

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
                                fp_out.write(line)

            if task == "qqp" and split != "test":
                with open(f"{processed_folder}/{split}.tsv.temp", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.tsv", "w", encoding="utf-8") as fp_out:
                        for line in fp_in:
                            if len(line.split("\t")) == 6:
                                fp_out.write(line)
            else:
                shutil.copy(f"{processed_folder}/{split}.tsv.temp", f"{processed_folder}/{split}.tsv")

        # Split into input0, input1 and label
        for split in splits:
            for input_type in range(input_count):
                if split != "test":
                    column_number = input_columns[input_type]
                else:
                    column_number = test_input_columns[input_type]
                with open(f"{processed_folder}/{split}.tsv", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.raw.input{input_type}.tsv", "w", encoding="utf-8") as fp_out:
                        for line in fp_in:
                            fp_out.write(line.split("\t")[column_number])
            if split != "test":
                with open(f"{processed_folder}/{split}.tsv", "r", encoding="utf-8") as fp_in:
                    with open(f"{processed_folder}/{split}.label", "w", encoding="utf-8") as fp_out:
                        col = dev_label_column if task == "mnli" and split != "train" else label_column
                        for line in fp_in:
                            fp_out.write(line.split("\t")[col])

            # BPE encode
            print(f"Running BPE encoding on '{task}' for '{split}' dataset")
            for input_type in range(input_count):
                lang = f"input{input_type}"
                os.system(
                    "python -m examples.roberta.multiprocessing_bpe_encoder "
                    f"--encoder-json {BPEMB_PATH}/encoder.json --vocab-bpe {BPEMB_PATH}/vocab.bpe "
                    f"--inputs {processed_folder}/{split}.raw.{lang} "
                    f"--outputs {processed_folder}/{split}.{lang} "
                    "--workers 16 --keep-empty"
                )

        if os.path.exists(f"{task}-bin"):
            shutil.rmtree(f"{task}-bin")

        devpref = f"{processed_folder}/dev.LANG"
        testpref = f"{processed_folder}/test.LANG"

        if task == "mnli":
            devpref = f"{processed_folder}/dev_matched.LANG,{processed_folder}/dev_mismatched.LANG"
            testpref = f"{processed_folder}/test_matched.LANG,{processed_folder}/test_mismatched.LANG"

        for input_type in range(input_count+1):
            lang = f"input{input_type}"
            os.system(
                f"fairseq-preprocess --only-source --trainpref {processed_folder}/train.label "
                f"--validpref {devpref.replace('LANG', '')}{lang} " +
                f"--validpref {testpref.replace('LANG', '')}{lang} " +
                f"--destdir {task}-bin/{lang} --workers 32 --srcdict {BPEMB_PATH}/dict.txt"
            )

        if task != "sts-b":
            os.system(
                f"fairseq-preprocess --only-source --trainpref {processed_folder}/train.label "
                f"--validpref {devpref.replace('LANG', '')}label " +
                f"--destdir {task}-bin/label --workers 32"
            )
        else:
            os.makedirs(f"{task}-bin/label")
            with open(f"{processed_folder}/train.label", "r", encoding="utf-8") as fp_in:
                with open(f"{task}-bin/label/train.label", "w", encoding="utf-8") as fp_out:
                    for line in fp_in:
                        fp_out.write(str(float(line.split("\t")[0]) / 5.0))
            with open(f"{processed_folder}/dev.label", "r", encoding="utf-8") as fp_in:
                with open(f"{task}-bin/label/valid.label", "w", encoding="utf-8") as fp_out:
                    for line in fp_in:
                        fp_out.write(str(float(line.split("\t")[0]) / 5.0))

        print(f"Done with '{task}'")

if __name__ == "__main__":
    if len(argv) < 2:
        print("Run as following:")
        print("./examples/roberta/preprocess_GLUE_tasks.sh <glud_data_folder> <task_name>")
        exit(1)

    preprocess_glue_task(argv[1])
