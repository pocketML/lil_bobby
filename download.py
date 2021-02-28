import os

def download_superglue():
    pass

TASK_DATASET_PATHS = {
    "superglue": "data/superglue",
    "superglue_ax_b": "data/superglue/ax_b",
    "superglue_ax_g": "data/superglue/ax_g",
    "superglue_bool_q": "data/superglue/bool_q",
    "superglue_cb": "data/superglue/cb",
    "superglue_copa": "data/superglue/copa",
    "superglue_multirc": "data/superglue/multirc",
    "superglue_record": "data/superglue/record",
    "superglue_rte": "data/superglue/rte",
    "superglue_wic": "data/superglue/wic",
    "superglue_wsc": "data/superglue/wsc",
    "squad": "data/squad"
}

DATASET_DOWNLOAD_FUNCS = {
    "superglue": download_superglue
}

def dataset_exists(path):
    return os.path.exists(path)

def download_data(task, path):
    DATASET_DOWNLOAD_FUNCS[task].download(path)

def get_dataset_path(task):
    path = TASK_DATASET_PATHS[task]
    if not dataset_exists(path):
        download_data(task, path)

    return path
