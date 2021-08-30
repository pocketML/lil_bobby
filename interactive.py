"""
This module is used for interacting directly with models through the command line.
"""
import torch

from common import argparsers, model_utils, task_utils
from compression.distillation import models as distill_models

def predict_roberta(model, sentence_data, device, task):
    """
    Get predictions from a finetuned RoBERTa model on a given GLUE task.

    Parameters
    ----------
    model : Finetuned RoBERTa model
        Finetuned RoBERTa model to use for predicting.
    sentence_data : Union[tuple[str], tuple[str, str]]
        Tuple of sentences to predict on. One sentence for SST-2, two for QQP/MNLI.
    device : torch.device
        Device on which to do the prediction.
    task : str
        GLUE task to predict on.

    Returns
    ----------
    tuple[list[float], str]
        Tuple of output logits and predicted label.
    """
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    encoded = model.encode(sentence_data)

    if task_utils.is_sentence_pair(task):
        x = encoded[0].to(device), encoded[1].to(device)
    else:
        x = encoded.to(device)

    pred_logits = model.predict('sentence_classification_head', x)
    pred_label = label_fn(pred_logits.argmax().item())

    return pred_logits.tolist(), pred_label

def predict_student(model, sentence_data, device, task):
    """
    Get predictions from a distilled student model on a given GLUE task.

    Parameters
    ----------
    model : Distilled student model
        Distilled student model used to predict with.
    sentence_data : Union[tuple[str], tuple[str, str]]
        Tuple of sentences to predict on. One sentence for SST-2, two for QQP/MNLI.
    device : torch.device
        Device on which to do the prediction.
    task : str
        GLUE task to predict on.

    Returns
    ----------
    tuple[list[float], str]
        Tuple of output logits and predicted label.
    """
    label_fn = lambda label: model.label_dict[str(label)]
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
    pred_label = label_fn(torch.max(pred_logits, 1)[1].item())

    return pred_logits.tolist(), pred_label

def main(args):
    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    is_roberta_model = model_utils.is_finetuned_model(args.arch)

    if is_roberta_model: # Load finetuned RoBERTa model.
        model_path = model_utils.get_model_path(args.task, "finetuned")
        model = model_utils.load_teacher(args.task, f"{model_path}/{args.model_name}", use_cpu=args.cpu)
    else: # Load distilled student model.
        model = distill_models.load_student(args.task, args.arch, not args.cpu, model_name=args.model_name)

    # Prepare model for inference.
    model.to(device)
    model.eval()

    while True:
        # Receive sentence data from stdin.
        sentence_data = input(">")
        if sentence_data in ("quit", "q", "exit"):
            break

        if task_utils.is_sentence_pair(args.task):
            # Sentences should be separated by § for sentence pairs tasks.
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
