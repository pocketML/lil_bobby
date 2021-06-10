from common import argparsers

import run_distill
import run_all_seeds

def main(args, args_remain):
    do_pruning = "--prune-magnitude" in args_remain or "--prune-topk" in args_remain
    do_quant = (
        "--ptq-embedding" in args_remain or "--dq-encoder" in args_remain or
        "--dq-classifier" in args_remain or "ptq-classifier" in args_remain
    )

    # Create a list of arguments for use with 'run_all_seeds.py'.
    args_list = []
    if "--prune-aware" not in args_remain:
        args_list = ["compress", "evaluate", "analyze", "--compression-actions"]

    if do_pruning:
        args_list.append("prune")
    if do_quant:
        args_list.append("quantize")

    args_list.extend(args_remain)

    # Add already specified args (task, alpha, student-arch, embed-type, embed-dim) to list.
    for key in args.__dict__:
        key_fmt = "--" + key.replace("_", "-")
        args_list.append(key_fmt)
        args_list.append(str(args.__dict__[key]))

    args_list.extend(["--theoretical-size"])

    if args.load_trained_model is not None:
        # Create a name for the experiment that we are running.
        name = args.load_trained_model

        if do_pruning:
            name += "_prune"
        if do_quant:
            name += "_quant"
        args_list.extend(["--name", name])

    if "--prune-aware" in args_remain:
        if args.load_trained_model is not None:
            print("Error: --load-trained-model should be None when doing pruning aware training.")
            exit(0)

        # Run pruning aware distillation with our current pruning/quantization arguments.
        args_distill, args_remain = argparsers.args_run_distill(args_list)
        run_distill.main(args_distill, args_remain)
        return
    else:
        args_list.extend(["--model-size", "--model-disk-size", "--transponder"])

    if args.load_trained_model is None:
        print("Error: --load-trained-model can't be None when compressing pre-trained model.")
        exit(0)

    final_args, args_remain = argparsers.args_run_all(args_list)

    run_all_seeds.main(final_args, args_remain)

if __name__ == "__main__":
    ARGS, REMAIN = argparsers.args_run_extra_compression()

    main(ARGS, REMAIN)
