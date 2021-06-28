import os
from omegaconf import base

import torch

from common import data_utils, model_utils, task_utils, argparsers
from compression.distillation import models as distill_models

MODEL_ARCHS = [
    "emb-ffn", "rnn", "bilstm", "large"
]

# "emb-ffn": "embffn_sst_alpha0_hash100_may18_hadfield",
# "rnn": "test_rrn_bigger",
# "bilstm": "tang_best",
# "large": "finetuned_sst-2_feynman"

def get_predictions_roberta(model, val_data):
    """
    Return all indices of sentences in the validation dataset
    that a finetuned roberta model predicted wrongly.
    """
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    predictions = []
    labels = []
    for x, target in zip(*val_data):
        preds = model.predict('sentence_classification_head', x).argmax().item()
        pred_label = label_fn(preds)
        target_label = label_fn(target.item())
        labels.append(target_label)
        predictions.append(pred_label)

    return predictions, labels

def get_predictions_distilled(model, val_data, task, device):
    """
    Return all indices of sentences in the validation dataset
    that a distilled model predicted wrongly.
    """
    predictions = []
    labels = []
    for x, lens, target_labels, _ in val_data:
        if task_utils.is_sentence_pair(task):
            x = x[0].to(device), x[1].to(device)
        else:
            x = x.to(device)
        target_labels = target_labels.to(device).squeeze()

        out_logits = model(x, lens)
        _, pred_labels = torch.max(out_logits, 1)
        labels.extend(target_labels.tolist())
        predictions.extend(pred_labels.tolist())

    return predictions, labels

def load_validation_data(model, task, is_roberta_model):
    val_data = data_utils.load_val_data(task)
    if is_roberta_model:
        data_x = []
        data_y = []
        for train_example in zip(*val_data):
            if task_utils.is_sentence_pair(task):
                x_1, target, x_2 = train_example
                x = x_1, x_2
            else:
                x_1, target = train_example
                x = (x_1,)
            encoded = model.encode(*x)

            label = torch.LongTensor([task_utils.TASK_LABEL_DICT[task][target.strip()]])

            data_x.append(encoded)
            data_y.append(label)
        return data_x, data_y

    return data_utils.get_val_dataloader(model, val_data, shuffle=False)

def get_predictions(args):
    model_names = {}
    for arch, model_name in zip(MODEL_ARCHS, args.model_names):
        model_names[arch] = model_name

    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    predicts_for_models = []

    print("Calculating and saving wrong predictions on validation data...")

    for arch in model_names:
        is_roberta_model = model_utils.is_finetuned_model(arch)
        model_name = model_names[arch]
        print(f"Arch: {arch} | Model: {model_name}")
        try:
            if is_roberta_model:
                model_path = model_utils.get_model_path(args.task, "finetuned")
                model = model_utils.load_teacher(args.task, f"{model_path}/{model_name}", use_cpu=args.cpu)
            else:
                model = distill_models.load_student(args.task, arch, not args.cpu, model_name=model_name)
                model.cfg["batch-size"] = args.batch_size
        except FileNotFoundError:
            print(f"Error: arch {arch} with name {model_names[arch]} does not exist.")
            print(f"Please input model-names in this order: {MODEL_ARCHS}.")

        data = load_validation_data(model, args.task, is_roberta_model)

        model.to(device)
        model.eval()

        if is_roberta_model:
            predicts, labels = get_predictions_roberta(model, data)
        else:
            predicts, labels = get_predictions_distilled(model, data, args.task, device)

        wrong = []
        correct = []
        for index, (pred, label) in enumerate(zip(predicts, labels)):
            list_to_add = correct if pred == label else wrong
            list_to_add.append(index)

        print(f"Number of wrong answers: {len(wrong)}/{len(predicts)}")

        predicts_for_models.append((correct, wrong))

    print("Writing data to disk.")

    for arch, predictions in zip(model_names, predicts_for_models):
        with open(f"misc/wrong_answers_{args.task}_{arch}.txt", "w", encoding="utf-8") as fp:
            for index in predictions[1]:
                fp.write(str(index) + "\n")

def load_predicted_sentences(task, val_sents):
    sentences_for_models = {}

    # For each model, load indexes of the sentences that the model classified incorrectly.
    for arch in MODEL_ARCHS:
        correct = []
        wrong = []
        with open(f"misc/wrong_answers_{task}_{arch}.txt", "r", encoding="utf-8") as fp:
            wrong_indices_set = set(int(x.strip()) for x in fp)
            for index in range(len(val_sents[0])):
                sents = (val_sents[0][index],)
                if task_utils.is_sentence_pair(task):
                    sent_2 = sents[1][index]
                    sents = (sents, sent_2)

                list_to_add = wrong if index in wrong_indices_set else correct
                list_to_add.append(sents)

        sentences_for_models[arch] = (correct, wrong)

    return sentences_for_models

def get_word_occurences(task, include_augmented=False):
    counts = {}
    base_folder = f'{task_utils.TASK_INFO[task]["path"]}'
    folders = [base_folder + "/processed/"]
    file_prefixes = ["train.raw"]
    if include_augmented:
        folders.append(base_folder + "/augment_data/")
        file_prefixes.append("tinybert")

    for file_prefix, folder in zip(file_prefixes, folders):
        files = [file_prefix + '.input0']
        if task_utils.is_sentence_pair(task):
            files.append(file_prefix + '.input1')

        # Count how often every word in the training set occured.
        for train_file in files:
            with open(folder + train_file, encoding='utf-8') as fp:
                for sentence in fp:
                    for word in sentence.split(" "):
                        counts[word] = counts.get(word, 0) + 1

    return counts

def analyze_sentence_lengths(model_sentences, task):
    """
    Return the average length of sentences that each model
    classified correctly and incorrectly, respectively.
    """
    avg_lengths = {}

    for arch in model_sentences:
        correct_sents, wrong_sents = model_sentences[arch]

        sum_len_wrong = 0
        sum_len_correct = 0

        # Go through all sentences in the validation dataset.
        # Sum lengths of sentences that were correctly and incorrectly classified.
        for correct_sent_sample in correct_sents:
            for sentence in correct_sent_sample: # Loop if sentence pair.
                sum_len_correct += len(sentence)

        for wrong_sent_sample in wrong_sents:
            for sentence in wrong_sent_sample: # Loop if sentence pair.
                sum_len_wrong += len(sentence)

        # Calculate total amount of correct/incorrect sentences.
        total_sentences_right = len(correct_sents)
        total_sentences_wrong = len(wrong_sents)
        if task_utils.is_sentence_pair(task):
            total_sentences_right *= 2
            total_sentences_wrong *= 2

        avg_lengths[arch] = (
            sum_len_correct / total_sentences_right,
            sum_len_wrong / total_sentences_wrong
        )
    return avg_lengths

def analyze_rare_words(model_sentences, word_counts):
    """
    Return the average occurence of words in the training set
    in validation sentences that each model classified
    correctly and incorrectly, respectively.
    """
    train_word_occurences = {}

    for arch in model_sentences:
        correct_sents, wrong_sents = model_sentences[arch]
    
        words_in_correct_sent = set()
        words_in_wrong_sent = set()

        # Create a set consisting of all words in sentences
        # that were correctly/incorrectly classified.
        for correct_sent_sample in correct_sents:
            for sentence in correct_sent_sample: # Loop if sentence pair.
                for word in sentence.split(" "):
                    words_in_correct_sent.add(word)

        for wrong_sent_sample in wrong_sents:
            for sentence in wrong_sent_sample: # Loop if sentence pair.
                for word in sentence.split(" "):
                    words_in_wrong_sent.add(word)

        sum_occurence_correct = 0
        sum_occurence_wrong = 0

        # Sum up how often each word in correct/incorrect sentences
        # occured in the training set.
        for word in words_in_correct_sent:
            sum_occurence_correct += word_counts.get(word, 0)
        for word in words_in_wrong_sent:
            sum_occurence_wrong += word_counts.get(word, 0)

        train_word_occurences[arch] = (
            sum_occurence_correct / len(words_in_correct_sent),
            sum_occurence_wrong / len(words_in_wrong_sent)
        )
    return train_word_occurences

def analyze_missing_words(model_sentences, word_counts):
    """
    Return the percentage of words in the validation set 
    that are not in the training set for sentences that
    each model classified correctly and incorrectly, respectively.
    """
    avg_occurences = {}

    for arch in model_sentences:
        correct_sents, wrong_sents = model_sentences[arch]

        words_in_correct_sent = set()
        words_in_wrong_sent = set()
        out_of_vocab_correct_sent = set()
        out_of_vocab_wrong_sent = set()

        # Create a set consisting of all words in sentences that
        # were correctly/incorrectly classified.
        for correct_sent_sample in correct_sents:
            for sentence in correct_sent_sample: # Loop if sentence pair.
                for word in sentence.split(" "):
                    words_in_correct_sent.add(word)
                    if word not in word_counts:
                        out_of_vocab_correct_sent.add(word)

        for wrong_sent_sample in wrong_sents:
            for sentence in wrong_sent_sample: # Loop if sentence pair.
                for word in sentence.split(" "):
                    words_in_wrong_sent.add(word)
                    if word not in word_counts:
                        out_of_vocab_wrong_sent.add(word)

        avg_occurences[arch] = (
            len(out_of_vocab_correct_sent) / len(words_in_correct_sent),
            len(out_of_vocab_wrong_sent) / len(words_in_wrong_sent)
        )
    return avg_occurences

def analyze_answers(task, val_sentences):
    sentences_for_models = load_predicted_sentences(task, val_sentences)

    sentence_lengths = analyze_sentence_lengths(sentences_for_models, task)

    for arch in sentence_lengths:
        print(f"---===--- {arch} ---===---")
        print(f"Avg. len. correct sents: {sentence_lengths[arch][0]}")
        print(f"Avg. len. wrong sents:   {sentence_lengths[arch][1]}")

    word_occurences = get_word_occurences(task)

    rare_words = analyze_rare_words(sentences_for_models, word_occurences)

    for arch in rare_words:
        print(f"---===--- {arch} ---===---")
        print(f"Occurence of words correct sents: {rare_words[arch][0]}")
        print(f"Occurence of words wrong sents:   {rare_words[arch][1]}")

    out_of_vocab = analyze_missing_words(sentences_for_models, word_occurences)

    for arch in out_of_vocab:
        print(f"---===--- {arch} ---===---")
        print(f"Missing words ratio correct sents: {out_of_vocab[arch][0]}")
        print(f"Missing words ratio wrong sents:   {out_of_vocab[arch][1]}")

def main(args):
    all_exists = True
    for arch in MODEL_ARCHS:
        if not os.path.exists(f"misc/wrong_answers_{args.task}_{arch}.txt"):
            all_exists = False
            break

    if not all_exists:
        if args.model_names is None or len(args.model_names) != 4:
            print(f"Error: please provide 4 trained model names in the order: {MODEL_ARCHS}")
            exit(0)

        get_predictions(args)
    else:
        print("*** Loading pre-computed predictions ***")

    val_data = data_utils.load_train_data(args.task, ds_type="dev")
    if task_utils.is_sentence_pair(args.task):
        sents_1, _, sents_2 = val_data
        sentences = [sents_1, sents_2]
    else:
        sents_1, _ = val_data
        sentences = [sents_1]
    sentences = [list(reversed([x.strip() for x in sents])) for sents in sentences]

    analyze_answers(args.task, sentences)

if __name__ == "__main__":
    ARGS = argparsers.args_task_difficulty()
    main(ARGS)
