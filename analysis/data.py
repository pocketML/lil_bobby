from common.task_utils import TASK_LABEL_DICT, is_sentence_pair

def get_label_distribution(data, task):
    label_count = {}
    label_index = 2 if is_sentence_pair(task) else 1
    for label_str in data[label_index]:
        label = TASK_LABEL_DICT[task][label_str]
        count = label_count.get(label, 0)
        label_count[label] = count + 1

    total_count = sum(label_count.values())
    return {x: label_count[x] / total_count for x in label_count}

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