import torch

from common import argparsers, model_utils, task_utils
from compression.distillation import models as distill_models

def predict_roberta(model, sentence_data, device, task):
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    encoded = model.encode(sentence_data)

    if task_utils.is_sentence_pair(task):
        x = encoded[0].to(device), encoded[1].to(device)
    else:
        x = sentence_data.to(device)

    pred_logits = model.predict('sentence_classification_head', x).argmax().item()
    pred_label = label_fn(pred_logits)

    return pred_logits, pred_label

def predict_student(model, sentence_data, device, task):
    if task_utils.is_sentence_pair(task):
        encoded_1 = torch.LongTensor(model.embedding.encode(sentence_data[0]))
        encoded_2 = torch.LongTensor(model.embedding.encode(sentence_data[1]))
        x = encoded_1.unsqueeze(0).to(device), encoded_2.unsqueeze(0).to(device)
        lens = torch.LongTensor([len(encoded_1)]), torch.LongTensor([len(encoded_2)])
    else:
        encoded = torch.LongTensor(model.embedding.encode(sentence_data))
        lens = torch.LongTensor([len(encoded)])
        x = encoded.unsqueeze(0).to(device)

    pred_logits = model(x, lens)
    _, pred_label = torch.max(pred_logits, 1)

    return pred_logits.tolist(), pred_label.item()

def main(args):
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    is_roberta_model = model_utils.is_finetuned_model(args.arch)

    if is_roberta_model:
        model_path = model_utils.get_model_path(args.task, "finetuned")
        model = model_utils.load_teacher(args.task, f"{model_path}/{args.model_name}", use_cpu=args.cpu)
    else:
        model = distill_models.load_student(args.task, args.arch, not args.cpu, model_name=args.model_name)

    model.to(device)
    model.eval()

    while True:
        sentence_data = input(">")
        if sentence_data in ("quit", "q", "exit"):
            break

        if task_utils.is_sentence_pair(args.task):
            if "§" not in sentence_data:
                print("Error: please input two sentences separated by '§' for sentence pair tasks.")
                continue
            sentence_data = tuple(x.strip() for x in sentence_data.split("§"))

        predict_func = predict_roberta if is_roberta_model else predict_student

        logits, label = predict_func(model, sentence_data, device, args.task)

        print(f"Logits: {logits} | Label: {label}")

if __name__ == "__main__":
    ARGS = argparsers.args_interactive()
    main(ARGS)
