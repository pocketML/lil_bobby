import argparse
import json
from datetime import datetime

def main(args):
    metrics_path = f"experiments/{args.name}/metrics.json"
    with open(metrics_path) as fp:
        metrics_data = json.load(fp)
        acc_metrics_map = {
            "eval_acc": "test.accuracy",
            "eval_f1": "test.f1",
            "eval_matched": "test.matched.accuracy",
            "eval_mismatched": "test.mismatched.accuracy"
        }
        for key in args.__dict__:
            value = getattr(args, key)
            if value is not None:
                sacred_key = acc_metrics_map.get(key, key)
                print(f"Setting {sacred_key} to {value}")
                time_now = datetime.now()
                metrics_entry = {
                    "steps": [0],
                    "timestamp": [time_now.strftime("%Y-%m-%dT%H:%M%S.%f")],
                    "values": [value]
                }
                metrics_data[sacred_key] = metrics_entry

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_data, fp, indent=2)

if __name__ == "__main__":
    AP = argparse.ArgumentParser()

    AP.add_argument("--name", type=str, required=True)
    AP.add_argument("--model-params", type=int)
    AP.add_argument("--nonzero-params", type=int)
    AP.add_argument("--model-disk-size", type=float)
    AP.add_argument("--theoretical-size", type=float)
    AP.add_argument("--eval-acc", type=float)
    AP.add_argument("--eval-f1", type=float)
    AP.add_argument("--eval-matched", type=float)
    AP.add_argument("--eval-mismatched", type=float)

    ARGS = AP.parse_args()

    main(ARGS)
