import os

import torch

from common import data_utils, model_utils, task_utils, argparsers
from compression.distillation import models as distill_models

MODEL_NAMES = {
    "emb-ffn": "embffn_sst_alpha0_hash100_may18_hadfield",
    "rnn": "test_rrn_bigger",
    "bilstm": "tang_best",
    "large": "finetuned_sst-2_feynman"
}

def get_wrong_predictions_roberta(model, val_data, task, device):
    """
    Return all indices of sentences in the validation dataset
    that a finetuned roberta model predicted wrongly.
    """
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    wrong_predictions = []
    index = 0
    for sent, target in zip(*val_data):
        if task_utils.is_sentence_pair(task):
            x = sent[0].to(device), sent[1].to(device)
        else:
            x = sent.to(device)
        target = target.to(device)
        preds = model.predict('sentence_classification_head', x).argmax().item()
        pred_label = label_fn(preds)
        target_label = label_fn(target.item())
        if pred_label != target_label:
            wrong_predictions.append(index)

        index += 1

    return wrong_predictions

def get_wrong_predictions_distilled(model, val_data, task, device):
    """
    Return all indices of sentences in the validation dataset
    that a distilled model predicted wrongly.
    """
    wrong_predictions = []
    batch_offset = 0
    for x, lens, target_labels, _ in val_data:
        if task_utils.is_sentence_pair(task):
            x = x[0].to(device), x[1].to(device)
        else:
            x = x.to(device)
        target_labels = target_labels.to(device).squeeze()

        out_logits = model(x, lens)
        _, pred_labels = torch.max(out_logits, 1)
        indices = torch.arange(batch_offset, batch_offset + len(pred_labels), 1)
        wrong_predicts = indices[pred_labels != target_labels].tolist()
        wrong_predictions.extend(wrong_predicts)

        batch_offset += x.shape[0]

    return wrong_predictions

def calculate_wrong_predictions(args):
    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    indices_for_models = []

    for arch in MODEL_NAMES:
        is_roberta_model = model_utils.is_finetuned_model(arch)
        model_name = MODEL_NAMES[arch]
        print(arch, model_name)
        if is_roberta_model:
            model_path = model_utils.get_model_path(args.task, "finetuned")
            model = model_utils.load_teacher(args.task, f"{model_path}/{model_name}", use_cpu=args.cpu)
        else:
            model = distill_models.load_student(args.task, arch, not args.cpu, model_name=model_name)
            model.cfg["batch-size"] = args.batch_size

        data = load_data(model, args.task, is_roberta_model)

        model.to(device)
        model.eval()

        if is_roberta_model:
            wrong_predict_indices = get_wrong_predictions_roberta(model, data, args.task, device)
        else:
            wrong_predict_indices = get_wrong_predictions_distilled(model, data, args.task, device)
        print(f"Number of wrong answers: {len(wrong_predict_indices)}")

        indices_for_models.append(wrong_predict_indices)

    print("Writing data to disk.")

    for arch, wrong_predict_indices in zip(MODEL_NAMES, indices_for_models):
        with open(f"misc/wrong_answers_{args.task}_{arch}.txt", "w", encoding="utf-8") as fp:
            for index in wrong_predict_indices:
                fp.write(str(index) + "\n")

def load_data(model, task, is_roberta_model):
    val_data = data_utils.load_val_data(task)
    if is_roberta_model:
        data_x = []
        data_y = []
        for train_example in zip(*val_data):
            if task_utils.is_sentence_pair(task):
                x_1, target, x_2 = train_example
                x = x_1, x_2
            else:
                x, target = train_example
            encoded = model.encode(x)

            label = torch.LongTensor([task_utils.TASK_LABEL_DICT[task][target.strip()]])

            data_x.append(encoded)
            data_y.append(label)
        return data_x, data_y

    return data_utils.get_val_dataloader(model, val_data, shuffle=False)

def analyze_sentence_lengths(task, wrong_sentences, all_sentences):
    """
    Return the average length of sentences that each model
    classified correctly and incorrectly, respectively.
    """
    avg_lengths = {}

    total_sentences = sum(len(sent_list) for sent_list in all_sentences)

    for arch in wrong_sentences:
        wrong_set = set()
        # Create a set of sentences that a model got wrong.
        for sent_data in wrong_sentences[arch]:
            if task_utils.is_sentence_pair(task):
                wrong_set.update(sent_data[0])
                wrong_set.update(sent_data[1])
            else:
                wrong_set.update(sent_data)

        sum_len_wrong = 0
        sum_len_correct = 0

        # Go through all sentences in the validation dataset.
        # Sum lengths of sentences that were correctly and incorrectly classified.
        for sent_list in all_sentences:
            for sentence in sent_list:
                if sentence not in wrong_set:
                    sum_len_correct += len(sentence)
                else:
                    sum_len_wrong += len(sentence)

        # Calculate total amount of correct/incorrect sentences.
        total_sentences_right = total_sentences - len(wrong_set)
        total_sentences_wrong = len(wrong_set)
        if task_utils.is_sentence_pair(task):
            total_sentences_right *= 2
            total_sentences_wrong *= 2

        avg_lengths[arch] = (
            sum_len_correct / total_sentences_right,
            sum_len_wrong / total_sentences_wrong
        )
    return avg_lengths

def analyze_rare_words(task, wrong_sentences, all_val_sentences):
    """
    Return the average occurence of words in the training set
    in validation sentences that each model classified
    correctly and incorrectly, respectively.
    """
    counts = {}
    folder = f'{task_utils.TASK_INFO[task]["path"]}/processed/'
    files = ['train.raw.input0']
    if task_utils.is_sentence_pair(task):
        files.append('train.raw.input1')

    # Count how often every word in the training set occured.
    for train_file in files:
        with open(folder + train_file, encoding='utf-8') as fp:
            for sentence in fp:
                for word in sentence:
                    counts[word] = counts.get(word, 0) + 1

    avg_occurences = {}

    for arch in wrong_sentences:
        wrong_set = set()
        # Create a set of sentences that a model got wrong.
        for sent_data in wrong_sentences[arch]:
            if task_utils.is_sentence_pair(task):
                wrong_set.update(sent_data[0])
                wrong_set.update(sent_data[1])
            else:
                wrong_set.update(sent_data)

        words_in_correct_sent = set()
        words_in_wrong_sent = set()

        # Create a set consisting of all words in sentences that
        # were correctly/incorrectly classified.
        for sent_list in all_val_sentences:
            for sentence in sent_list:
                for word in sentence.split(" "):
                    if sentence in wrong_set:
                        words_in_correct_sent.add(word)
                    else:
                        words_in_wrong_sent.add(word)

        sum_occurence_correct = 0
        sum_occurence_wrong = 0

        # Sum up how often each word in correct/incorrect sentences
        # occured in the training set.
        for word in words_in_correct_sent:
            sum_occurence_correct += counts.get(word, 0)
        for word in words_in_wrong_sent:
            sum_occurence_wrong += counts.get(word, 0)

        avg_occurences[arch] = (
            sum_occurence_correct / len(words_in_correct_sent),
            sum_occurence_wrong / len(words_in_wrong_sent)
        )
    return avg_occurences

def analyze_answers(task, sentences, labels):
    indices_for_models = []
    # For each model, load indexes of the sentences that the model classified incorrectly.
    for arch in MODEL_NAMES:
        with open(f"misc/wrong_answers_{task}_{arch}.txt", "r", encoding="utf-8") as fp:
            indices_for_models.append([int(x.strip()) for x in fp])

    indices_for_models.sort(key=lambda x: len(x), reverse=True)

    sentences_for_models = {}

    # Load validation sentences and labels for the sentences that each model got wrong.
    for arch, model_indices in zip(MODEL_NAMES, indices_for_models):
        model_sentences = []
        for index in model_indices:
            sents = sentences[0][index]
            if len(sentences) == 2:
                sent_2 = sentences[1][index]
                sents = (sents, sent_2)

            model_sentences.append((sents, labels[index]))
        sentences_for_models[arch] = model_sentences

    sentence_lengths = analyze_sentence_lengths(task, sentences_for_models, sentences)

    for arch in sentence_lengths:
        print(f"---===--- {arch} ---===---")
        print(f"Avg. len. correct sents: {sentence_lengths[arch][0]}")
        print(f"Avg. len. wrong sents:   {sentence_lengths[arch][1]}")

    rare_words = analyze_rare_words(task, sentences_for_models, sentences)

    for arch in rare_words:
        print(f"---===--- {arch} ---===---")
        print(f"Avg. occurence of words correct sents: {rare_words[arch][0]}")
        print(f"Avg. occurence of words wrong sents:   {rare_words[arch][1]}")

    # shared_wrong_answers = []
    # for count in range(1, len(sentences_for_models)):
    #     common_wrong = set(sentences_for_models[0])
    #     for model_sentences in sentences_for_models[:-count]:
    #         common_wrong = set.intersection(common_wrong, set(model_sentences))
    #     shared_wrong_answers.append(common_wrong)

    # print("=== Sentences every model got wrong ====")
    # for sentence, label in shared_wrong_answers[0]:
    #     print(f"{sentence} ({label})")

    # print()

    # print("=== Sentences student models got wrong ====")
    # for sentence, label in shared_wrong_answers[1]:
    #     print(f"{sentence} ({label})")

    # print(f"Shared across all: {len(shared_wrong_answers[0])}")
    # print(f"Shared across students: {len(shared_wrong_answers[1])}")
    # print(f"Shared across FFN + RNN: {len(shared_wrong_answers[2])}")

def main(args):
    all_exists = True
    for arch in MODEL_NAMES:
        if not os.path.exists(f"misc/wrong_answers_{args.task}_{arch}.txt"):
            all_exists = False
            break

    if not all_exists:
        calculate_wrong_predictions(args)

    val_data = data_utils.load_train_data(args.task, ds_type="dev")
    if task_utils.is_sentence_pair(args.task):
        sents_1, labels, sents_2 = val_data
        sentences = [sents_1, sents_2]
    else:
        sents_1, labels = val_data
        sentences = [sents_1]
    sentences = [list(reversed([x.strip() for x in sents])) for sents in sentences]
    labels = list(reversed([x.strip() for x in labels]))

    analyze_answers(args.task, sentences, labels)

if __name__ == "__main__":
    ARGS = argparsers.args_task_difficulty()
    main(ARGS)
