"""
This module is used for doing various preprocessing tasks.
These include preprocessing GLUE task data, running data augmentation,
and generating distillation loss for RoBERTa model.
"""

import torch

from common import argparsers, model_utils, task_utils
from preprocessing import download, data_augment, distillation_loss

def main(args):
    # Set seed for all PyTorch functions.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.glue_preprocess: # Run GLUE preprocessing.
        download_url = task_utils.TASK_INFO[args.task]['download_url']
        target_folder = task_utils.TASK_INFO[args.task]["path"]
        # Download data and preprocess it.
        download.download_and_process_data(args.task, download_url, target_folder)
    elif args.augment is not None: # Run data augmentation.
        data_augment.augment(args.task, args.augment, args.seed)
    elif args.generate_loss is not None: # Generate distillation loss using teacher model.
        if args.model_name is None:
            raise ValueError("Model name is required for generate_loss!")

        model_path = model_utils.get_model_path(args.task, "finetuned")
        teacher_model = model_utils.load_teacher(
            args.task, f"{model_path}/{args.model_name}", args.cpu
        )
        distillation_loss.generate_distillation_loss(args, teacher_model)

if __name__ == "__main__":
    ARGS = argparsers.args_preprocess()
    main(ARGS)