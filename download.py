import os
import shutil
import requests
import argparsers

# *===================================================================*
# *                          DOWNLOAD URLS                            *
# *===================================================================*
SUPERGLUE_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset"

DATASET_DOWNLOAD_URLS = {
    "superglue": [f"{SUPERGLUE_URL}/combined.zip"],
    "superglue_ax-b": [f"{SUPERGLUE_URL}/AX-b.zip"],
    "superglue_cb": [f"{SUPERGLUE_URL}/CB.zip"],
    "superglue_mult": [f"{SUPERGLUE_URL}/COPA.zip"],
    "superglue_multirc": [f"{SUPERGLUE_URL}/MultiRC.zip"],
    "superglue_rte": [f"{SUPERGLUE_URL}/RTE.zip"],
    "superglue_wic": [f"{SUPERGLUE_URL}/WiC.zip"],
    "superglue_wsc": [f"{SUPERGLUE_URL}/WS.zipC"],
    "superglue_boolq": [f"{SUPERGLUE_URL}/BoolQ.zip"],
    "superglue_record": [f"{SUPERGLUE_URL}/ReCoRD.zip"],
    "superglue_ax-g": [f"{SUPERGLUE_URL}/AX-g.zip"],
    "squad": [f"{SQUAD_URL}/train-v2.0.json", f"{SQUAD_URL}/dev-v2.0.json"]
}

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

        with open(filename, "wb") as fp:
            for chunk in response.iter_content(chunk_size=128):
                fp.write(chunk)

        if "json" in filetype:
            if not os.path.exists(folder):
                os.mkdir(folder)
            shutil.move(filename, f"{folder}/{filename}")
        else: # File is an archive.
            print(base_folder)
            print(filename)
            shutil.unpack_archive(filename, base_folder)

            os.remove(filename)

        print(f"Downloaded '{url}'")

def path_exists(folder):
    return os.path.exists(folder)

def get_dataset_path(task):
    folder = TASK_DATASET_PATHS[task]
    if not path_exists(folder):
        urls = DATASET_DOWNLOAD_URLS[task]
        download_and_extract(urls, folder)

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
        download_urls = DATASET_DOWNLOAD_URLS[args.task]
        download_and_extract(download_urls, target_folder)
    else:
        target_folder = MODEL_PATHS[args.model]
        download_urls = MODEL_DOWNLOAD_URLS[args.model]
        download_and_extract(download_urls, target_folder)
