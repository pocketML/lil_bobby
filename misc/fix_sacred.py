import argparse
import json
from datetime import datetime

def main(args):
    metrics_path = f"experiments/{args.name}/metrics.json"
    with open(metrics_path) as fp:
        metrics_data = json.load(fp)
        keys = [
            "params", "nonzero_params",
            "model_disk_size",
            "theoretical_size", "eval_acc"
        ]
        for key in keys:
            if key is not None:
                value = getattr(args, key)
                print(f"Setting {key} to {value}")
                time_now = datetime.now()
                metrics_entry = {
                    "steps": [0],
                    "timestamp": [time_now.strftime("%Y-%m-%dT%H:%M%S.%f")],
                    "values": [value]
                }
                metrics_data[key] = metrics_entry

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_data, fp, indent=2)

if __name__ == "__main__":
    AP = argparse.ArgumentParser()

    AP.add_argument("--name", type=str, required=True)
    AP.add_argument("--params", type=int)
    AP.add_argument("--nonzero-params", type=int)
    AP.add_argument("--model-disk-size", type=float)
    AP.add_argument("--theoretical-size", type=float)
    AP.add_argument("--eval-acc", type=float)

    ARGS = AP.parse_args()

    main(ARGS)
