import os
import shutil
import requests
from preprocessing import preprocess_GLUE_tasks

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

GLUE_TASKS = ["mnli", "qnli", "qqp", "rte", "sst-2", "mrpc", "cola", "sts-b", "ax"]

# *===================================================================*
# *                            DATA PATHS                             *
# *===================================================================*
MODEL_PATHS = {
    "base": "models/base",
    "large": "models/large"
}

def download_and_extract(urls, folder):
    base_folder = "/".join(folder.split("/")[:-1]) + "/"
    os.makedirs(base_folder, exist_ok=True)

    for url in urls:
        print(f"Downloading '{url}' to '{base_folder}'...", flush=True)
        filename = os.path.basename(url)
        filetype = filename.split(".")[-1]
        response = requests.get(url)

        with open(filename, "wb+") as fp:
            for chunk in response.iter_content(chunk_size=128):
                fp.write(chunk)

        if "json" in filetype or "tsv" in filetype or "txt" in filetype:
            os.makedirs(folder, exist_ok=True)
            shutil.move(filename, f"{folder}/{filename}")
        else: # File is an archive.
            shutil.unpack_archive(filename, base_folder)

            os.remove(filename)


def preprocess_glue_task(task):
    preprocess_GLUE_tasks.preprocess_glue_task(task)

def path_exists(folder):
    return os.path.exists(folder)

def download_and_process_data(task, urls, folder):
    download_and_extract(urls, folder)
    if task in GLUE_TASKS: # Glue task needs preprocessing.
        preprocess_glue_task(task)

def get_dataset_path(task, task_info):
    folder = task_info["path"]
    if not path_exists(folder):
        urls = task_info["download_url"]
        download_and_process_data(task, urls, folder)

    return folder

def get_roberta_path(model_info):
    folder = model_info["path"]
    if not path_exists(folder):
        urls = model_info["download_url"]
        download_and_extract(urls, folder)

    return folder
