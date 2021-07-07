from common import argparsers, task_utils, data_utils
from analysis import data as data_analysis

def main(args):
    og_path = f'{task_utils.TASK_INFO[args.task]["path"]}/distillation_data/train.tsv'
    og_data = data_utils.load_distillation_data(og_path)
    og_labels = data_analysis.get_label_distribution(og_data, args.task)

    augment_path = f'{task_utils.TASK_INFO[args.task]["path"]}/distillation_data/tinybert.tsv'
    #augment_data = data_utils.load_distillation_data(augment_path)
    augment_data = data_utils.load_downsampled_distillation_data(args.task, augment_path, og_labels)
    augment_labels = data_analysis.get_label_distribution(augment_data, args.task)

    print(f"Label distribution (original): {og_labels}")
    print(f"Label distribution (augment): {augment_labels}")

    words_augment = data_analysis.get_word_distribution(augment_data, args.task)
    words_og = data_analysis.get_word_distribution(og_data, args.task)

    total_count_og = sum([x[1] for x in words_og])
    total_count_aug = sum([x[1] for x in words_augment])

    og_print = [(x[0], f'{(x[1] / total_count_og)*100:.6f}%') for x in words_og[:25]]
    aug_print = [(x[0], f'{(x[1] / total_count_aug)*100:.6f}%') for x in words_augment[:25]]

    print(f"Word distribution (original): {og_print}")
    print(f"Word distribution (augment): {aug_print}")

if __name__ == "__main__":
    ARGS = argparsers.args_validate_augment()
    main(ARGS)
