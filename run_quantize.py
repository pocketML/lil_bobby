import argparse

from misc.load_results import EXTRA_COMPRESSION_MODELS
import run_extra_compression
from common import argparsers

ap = argparse.ArgumentParser()

ap.add_argument("--task", choices=("sst-2", "qqp", "mnli"))
ap.add_argument("--arch", choices=("bilstm", "rnn", "emb-ffn"))
ap.add_argument("--id", choices=("a", "b", "c"))

args = ap.parse_args()

arch_map = {"bilstm": 0, "rnn": 3, "emb-ffn": 6}
id_map = {"a": 0, "b": 1, "c": 2}
task_map = {"sst-2": 0, "qqp": 1, "mnli": 2}

arch_id = arch_map[args.arch]
model_id = id_map[args.id]
task_id = task_map[args.task]

model_index = arch_id + model_id

model_name = EXTRA_COMPRESSION_MODELS[model_index][task_id]

command = (
    f"python run_extra_compression.py --task {args.task} --student-arch {args.arch} " +
    f"--ptq-embedding --dq-encoder --dq-classifier --load-trained-model {model_name}"
)

args_list = [
    "--task", args.task, "--student-arch", args.arch,
    "--ptq-embedding", "--dq-encoder", "--dq-classifier",
    "--load-trained-model", model_name
]

args, args_remain = argparsers.args_run_extra_compression(args_list)

run_extra_compression.main(args, args_remain)
