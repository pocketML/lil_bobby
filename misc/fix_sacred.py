import argparse
import json

def main(args):
    metrics_path = f"experiments/{args.name}/metrics.json"
    with open(metrics_path) as fp:
        metrics_data = json.load(fp)
        keys = [
            args.params, args.nonzero_params, args.model_disk_size,
            args.theoretical_size, args.eval_acc
        ]
        for key in keys:
            if key is not None:
                value = getattr(args, key)
                print(f"Setting {key} to {value}")
                metrics_data[key] = value

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_data, fp, indent=2)

if __name__ == "__main__":
    AP = argparse.ArgumentParser()

    AP.add_argument("--name", type=str, required=True)
    AP.add_argument("--params")
    AP.add_argument("--nonzero-params")
    AP.add_argument("--model-disk-size")
    AP.add_argument("--theoretical-size")
    AP.add_argument("--eval-acc")

    ARGS = AP.parse_args()

    main(ARGS)
