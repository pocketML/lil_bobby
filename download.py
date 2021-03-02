import os
import shutil
import requests
import argparsers
from misc import preprocess_GLUE_tasks

# *===================================================================*
# *                          DOWNLOAD URLS                            *
# *===================================================================*
SUPERGLUE_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2"
GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset"

GLUE_DOWNLOAD_URLS = {
    "glue_cola": [f"{GLUE_URL}/CoLA.zip"],
    "glue_sst": [f"{GLUE_URL}/SST-2.zip"],
    "glue_sts": [f"{GLUE_URL}/STS-B.zip"],
    "glue_qqp": [f"{GLUE_URL}/QQP.zip"],
    "glue_mrpc": [f"{GLUE_URL}/MRPC.zip"],
    "glue_mnli": [f"{GLUE_URL}/MNLI.zip"],
    "glue_qnli": [f"{GLUE_URL}/QNLI.zip"],
    "glue_rte": [f"{GLUE_URL}/RTE.zip"],
    "glue_wnli": [f"{GLUE_URL}/WNLI.zip"],
    "glue_ax": [f"{GLUE_URL}/AX.tsv"]
}

DATASET_DOWNLOAD_URLS = {
    "superglue": [f"{SUPERGLUE_URL}/combined.zip"],
    "superglue_ax-b": [f"{SUPERGLUE_URL}/AX-b.zip"],
    "superglue_cb": [f"{SUPERGLUE_URL}/CB.zip"],
    "superglue_mult": [f"{SUPERGLUE_URL}/COPA.zip"],
    "superglue_multirc": [f"{SUPERGLUE_URL}/MultiRC.zip"],
    "superglue_rte": [f"{SUPERGLUE_URL}/RTE.zip"],
    "superglue_wic": [f"{SUPERGLUE_URL}/WiC.zip"],
    "superglue_wsc": [f"{SUPERGLUE_URL}/WSC.zip"],
    "superglue_boolq": [f"{SUPERGLUE_URL}/BoolQ.zip"],
    "superglue_record": [f"{SUPERGLUE_URL}/ReCoRD.zip"],
    "superglue_ax-g": [f"{SUPERGLUE_URL}/AX-g.zip"],
    "squad": [f"{SQUAD_URL}/train-v2.0.json", f"{SQUAD_URL}/dev-v2.0.json"],
    "glue": [x[0] for x in GLUE_DOWNLOAD_URLS.values()]
}

DATASET_DOWNLOAD_URLS.update(GLUE_DOWNLOAD_URLS)

MODEL_DOWNLOAD_URLS = {
    "base": ["https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz"],
    "large": ["https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz"]
}

# *===================================================================*
# *                            DATA PATHS                             *
# *===================================================================*
TASK_DATASET_PATHS = {
    "superglue": "data/superglue/all",
    "superglue_ax-b": "data/superglue/AX-b",
    "superglue_cb": "data/superglue/CB",
    "superglue_copa": "data/superglue/COPA",
    "superglue_multirc": "data/superglue/MultiRC",
    "superglue_rte": "data/superglue/RTE",
    "superglue_wic": "data/superglue/WiC",
    "superglue_wsc": "data/superglue/WSC",
    "superglue_boolq": "data/superglue/BoolQ",
    "superglue_record": "data/superglue/ReCoRD",
    "superglue_ax-g": "data/superglue/AX-g",
    "glue": "data/glue/all",
    "glue_cola": "data/glue/CoLA",
    "glue_sst": "data/glue/SST-2",
    "glue_sts": "data/glue/STS-B",
    "glue_qqp": "data/glue/QQP",
    "glue_mrpc": "data/glue/MRPC",
    "glue_mnli": "data/glue/MNLI",
    "glue_qnli": "data/glue/QNLI",
    "glue_rte": "data/glue/RTE",
    "glue_wnli": "data/glue/WNLI",
    "glue_ax": "data/glue/AX",
    "squad": "data/squad"
}

MODEL_PATHS = {
    "base": "models/base",
    "large": "models/large"
}

def download_and_extract(urls, folder):
    base_folder = "/".join(folder.split("/")[:-1]) + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    for url in urls:
        filename = os.path.basename(url)
        filetype = filename.split(".")[-1]
        response = requests.get(url)

        with open(filename, "wb+") as fp:
            for chunk in response.iter_content(chunk_size=128):
                fp.write(chunk)

        if "json" in filetype or "tsv" in filetype:
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.move(filename, f"{folder}/{filename}")
        else: # File is an archive.
            shutil.unpack_archive(filename, base_folder)

            os.remove(filename)

        print(f"Downloaded '{url}' to '{base_folder}'")

def preprocess_glue_task(task):
    preprocess_GLUE_tasks.preprocess_glue_task(task)

def path_exists(folder):
    return os.path.exists(folder)

def download_and_process_data(task, folder):
    urls = DATASET_DOWNLOAD_URLS[task]
    download_and_extract(urls, folder)
    if task in GLUE_DOWNLOAD_URLS:
        preprocess_glue_task(task)

def get_dataset_path(task):
    folder = TASK_DATASET_PATHS[task]
    if not path_exists(folder):
        download_and_process_data(task, folder)

    return folder

def get_model_path(model_type):
    folder = MODEL_PATHS[model_type]
    if not path_exists(folder):
        urls = MODEL_DOWNLOAD_URLS[model_type]
        download_and_extract(urls, folder)

    return folder

if __name__ == "__main__":
    args = argparsers.args_download()

    if args.task is not None:
        target_folder = TASK_DATASET_PATHS[args.task]
        preprocess_glue_task(args.task)
        #download_and_process_data(args.task, target_folder)
    else:
        target_folder = MODEL_PATHS[args.model]
        download_urls = MODEL_DOWNLOAD_URLS[args.model]
        download_and_extract(download_urls)
