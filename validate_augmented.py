from common import argparsers
from common.task_utils import TASK_INFO, TASK_LABEL_DICT, is_sentence_pair
from common.data_utils import load_distillation_data

def get_label_distribution(data, task):
    label_count = {}
    label_index = 2 if is_sentence_pair(task) else 1
    for label_str in data[label_index]:
        label = TASK_LABEL_DICT[task][label_str]
        count = label_count.get(label, 0)
        label_count[label] = count + 1

    total_count = sum(label_count.values())
    return {x: f"{int((label_count[x] / total_count) * 100)}%" for x in label_count}

def get_word_distribution(data, task):
    word_count = {}

    sentences = 2 if is_sentence_pair(task) else 1
    for sent_index in range(sentences):
        for sentence in data[sent_index]:
            words = sentence.split(" ")
            for word in words:
                count = word_count.get(word, 0)
                word_count[word] = count + 1

    pairs = list(word_count.items())
    pairs.sort(key=lambda x: x[1], reverse=True)

    return pairs

def main(args):
    path = f'{TASK_INFO[args.task]["path"]}/distillation_data/tinybert.tsv'
    augment_data = load_distillation_data(path)
    path = f'{TASK_INFO[args.task]["path"]}/distillation_data/train.tsv'
    og_data = load_distillation_data(path)

    labels_augment = get_label_distribution(augment_data, args.task)
    labels_og = get_label_distribution(og_data, args.task)

    print(f"Label distribution (original): {labels_og}")
    print(f"Label distribution (augment): {labels_augment}")

    words_augment = get_word_distribution(augment_data, args.task)
    words_og = get_word_distribution(og_data, args.task)

    print(f"Word distribution (original): {words_og[:25]}")
    print(f"Word distribution (augment): {words_augment[:25]}")

if __name__ == "__main__":
    ARGS = argparsers.args_validate_augment()
    main(ARGS)
