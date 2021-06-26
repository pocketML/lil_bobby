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

def analyze_sentence_lengths(wrong_sentences, all_sentences):
    avg_lengths = {}
    
    for arch in wrong_sentences:
        wrong_set = set(x[0] for x in wrong_sentences[arch])
        sum_len_wrong = 0
        sum_len_correct = 0
        for sentence in all_sentences:
            if sentence not in wrong_set:
                sum_len_correct += len(sentence)
            else:
                sum_len_wrong += len(sentence)
        avg_lengths[arch] = (
            sum_len_correct / (len(all_sentences) - len(wrong_set)),
            sum_len_wrong / len(wrong_set)
        )
    return avg_lengths

def analyze_rare_words(task, wrong_sentences, all_sentences):
    counts = {}
    for sentence in all_sentences:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1

    sum_occurence_correct = 0
    sum_occurence_wrong = 0


def analyze_answers(task, sentences, labels):
    indices_for_models = []
    for arch in MODEL_NAMES:
        with open(f"misc/wrong_answers_{task}_{arch}.txt", "r", encoding="utf-8") as fp:
            indices_for_models.append([int(x.strip()) for x in fp])

    indices_for_models.sort(key=lambda x: len(x), reverse=True)

    sentences_for_models = {}

    for arch, model_indices in zip(MODEL_NAMES, indices_for_models):
        model_sentences = []
        for index in model_indices:
            model_sentences.append((sentences[index], labels[index]))
        sentences_for_models[arch] = model_sentences

    sentence_lengths = analyze_sentence_lengths(sentences_for_models, sentences)
    
    for arch in sentence_lengths:
        print(f"---===--- {arch} ---===---")
        print(f"Avg. len. correct sents: {sentence_lengths[arch][0]}")
        print(f"Avg. len. wrong sents:   {sentence_lengths[arch][1]}")

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
    # Idea:
    # 1. Load all data for a task.
    # 2. Load every trained model (even finetuned). Start with worst model (FFN).
    # 3. For every model, run through every data point, add all wrong predictions to a list.
    # 4. See which questions all the 'wrong lists' have in common. 
    # 5. What are characteristics of these questions? Why are they hard?
    # 5. Look at lengths of sentences they get wrong vs. right.
    # 5. Look at whether there are rare/out of vocab words in sentences they get wrong.
    # 5. Look at whether specific words decrease accuracy in a meaningful way.
    # 6. ???
    # 7. Profit!
    all_exists = True
    for arch in MODEL_NAMES:
        if not os.path.exists(f"misc/wrong_answers_{args.task}_{arch}.txt"):
            all_exists = False
            break

    if not all_exists:
        calculate_wrong_predictions(args)

    sentences, labels = data_utils.load_train_data(args.task, ds_type="dev")
    sentences = list(reversed([x.strip() for x in sentences]))
    labels = list(reversed([x.strip() for x in labels]))

    analyze_answers(args.task, sentences, labels)

if __name__ == "__main__":
    ARGS = argparsers.args_task_difficulty()
    main(ARGS)
