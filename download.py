import os
import shutil
import requests
from common import argparsers
from misc import preprocess_GLUE_tasks
from common.task_utils import TASK_INFO
from common.model_utils import MODEL_INFO

# *===================================================================*
# *                          DOWNLOAD URLS                            *
# *===================================================================*
SUPERGLUE_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2"
GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset"

# DATASET_DOWNLOAD_URLS = {
#     "superglue": [f"{SUPERGLUE_URL}/combined.zip"],
#     "superglue_ax-b": [f"{SUPERGLUE_URL}/AX-b.zip"],
#     "superglue_cb": [f"{SUPERGLUE_URL}/CB.zip"],
#     "superglue_mult": [f"{SUPERGLUE_URL}/COPA.zip"],
#     "superglue_multirc": [f"{SUPERGLUE_URL}/MultiRC.zip"],
#     "superglue_rte": [f"{SUPERGLUE_URL}/RTE.zip"],
#     "superglue_wic": [f"{SUPERGLUE_URL}/WiC.zip"],
#     "superglue_wsc": [f"{SUPERGLUE_URL}/WSC.zip"],
#     "superglue_boolq": [f"{SUPERGLUE_URL}/BoolQ.zip"],
#     "superglue_record": [f"{SUPERGLUE_URL}/ReCoRD.zip"],
#     "superglue_ax-g": [f"{SUPERGLUE_URL}/AX-g.zip"],
#     "squad": [f"{SQUAD_URL}/train-v2.0.json", f"{SQUAD_URL}/dev-v2.0.json"],
#     "glue": [x[0] for x in GLUE_DOWNLOAD_URLS.values()]
# }

# Add download urls from TASK_INFO dictionary.
DATASET_DOWNLOAD_URLS = {
    task: TASK_INFO[task]["download_url"]
    for task in TASK_INFO
}

GLUE_TASKS = ["mnli", "qnli", "qqp", "rte", "sst-2", "mrpc", "cola", "sts-b", "ax"]

# Add 'glue' key that downloads all glue tasks in one go.
DATASET_DOWNLOAD_URLS["glue"] = [
    TASK_INFO[task]["download_url"]
    for task in GLUE_TASKS
]

MODEL_DOWNLOAD_URLS = {
    model: MODEL_INFO[model]["download_url"]
    for model in MODEL_INFO
}

# *===================================================================*
# *                            DATA PATHS                             *
# *===================================================================*
MODEL_PATHS = {
    "base": "models/base",
    "large": "models/large"
}

def download_and_extract(urls, folder):
    base_folder = "/".join(folder.split("/")[:-1]) + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    for url in urls:
        print(f"Downloading '{url}' to '{base_folder}'...", flush=True)
        filename = os.path.basename(url)
        filetype = filename.split(".")[-1]
        response = requests.get(url)

        with open(filename, "wb+") as fp:
            for chunk in response.iter_content(chunk_size=128):
                fp.write(chunk)

        if "json" in filetype or "tsv" in filetype or "txt" in filetype:
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.move(filename, f"{folder}/{filename}")
        else: # File is an archive.
            shutil.unpack_archive(filename, base_folder)

            os.remove(filename)


def preprocess_glue_task(task):
    preprocess_GLUE_tasks.preprocess_glue_task(task)

def path_exists(folder):
    return os.path.exists(folder)

def download_and_process_data(task, folder):
    urls = DATASET_DOWNLOAD_URLS[task]
    download_and_extract(urls, folder)
    if task in GLUE_TASKS: # Glue task needs preprocessing.
        preprocess_glue_task(task)

def get_dataset_path(task):
    folder = TASK_INFO[task]["path"]
    if not path_exists(folder):
        download_and_process_data(task, folder)

    return folder

def get_model_path(model_type):
    folder = MODEL_INFO[model_type]["path"]
    if not path_exists(folder):
        urls = MODEL_DOWNLOAD_URLS[model_type]
        download_and_extract(urls, folder)

    return folder

if __name__ == "__main__":
    ARGS = argparsers.args_download()

    if ARGS.task is not None:
        TARGET_FOLDER = TASK_INFO[ARGS.task]["path"]
        #preprocess_glue_task(ARGS.task)
        download_and_process_data(ARGS.task, TARGET_FOLDER)
    else:
        TARGET_FOLDER = MODEL_INFO[ARGS.model]["path"]
        DOWNLOAD_URLS = MODEL_DOWNLOAD_URLS[ARGS.model]
        download_and_extract(DOWNLOAD_URLS, TARGET_FOLDER)
