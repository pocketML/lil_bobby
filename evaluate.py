from time import time

from common import argparsers, task_utils, model_utils, data_utils
from compression.distillation import models as distill_models

from tqdm import tqdm
import torch

# for finetuned RobertaModel
def prepare_eval_data(model, task, filepath):
    eval_data = []
    with open(filepath, encoding="utf-8") as fin:
        fin.readline()
        for i, line in enumerate(fin):
            tokens = line.strip().split('\t')
            if len(tokens) == 0:
                continue
            if task == 'sst-2':
                sent, target = tokens[0], tokens[1]
                encoded = model.encode(sent)
            elif task == 'rte':
                sent1, sent2, target = tokens[1], tokens[2], tokens[3]
                encoded = model.encode(sent1, sent2)
            elif task == 'qqp':
                sent1, sent2, target = tokens[3], tokens[4], tokens[5]
                encoded = model.encode(sent1, sent2)
            elif task == 'mnli':
                sent1, sent2, target = tokens[8], tokens[9], tokens[15]
                encoded = model.encode(sent1, sent2)
            eval_data.append((encoded, target))
    return eval_data

def update_f1_counts_roberta(pred_label, target, tp, fp, fn):
    tp += int(pred_label == '1' and target == '1')
    fp += int(pred_label == '1' and target == '0')
    fn += int(pred_label == '0' and target == '1')
    return tp, fp, fn

# f1 is the harmonic mean of the precision and recall
# only for roberta_model at the moment
def evaluate_accuracy(model, task, val_data_path, include_f1=False):
    eval_data = prepare_eval_data(model, task, val_data_path)
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    ncorrect, tp, fp, fn = 0, 0, 0, 0
    for encoded, target in eval_data:
        pred = model.predict('sentence_classification_head', encoded).argmax().item()
        pred_label = label_fn(pred)
        ncorrect += int(pred_label == target)
        if include_f1:
            tp, fp, fn = update_f1_counts_roberta(pred_label, target, tp, fp, fn)

    accuracy = ncorrect / len(eval_data)
    if include_f1:
        f1_score = tp / (tp + 0.5 * (fp + fn))
        return (accuracy, f1_score)
    else:
        return accuracy

def update_f1_counts_distilled(pred_labels, target_labels, tp, fp, fn, tn):
    pred_ones = pred_labels == 1
    target_ones = target_labels == 1
    pred_zeros = pred_labels == 0
    target_zeros = target_labels == 0
    tp += torch.count_nonzero(torch.logical_and(pred_ones, target_ones)).item()
    fp += torch.count_nonzero(torch.logical_and(pred_ones, target_zeros)).item()
    fn += torch.count_nonzero(torch.logical_and(pred_zeros, target_ones)).item()
    tn += torch.count_nonzero(torch.logical_and(pred_zeros, target_zeros)).item()
    return tp, fp, fn, tn

def evaluate_distilled_model(model, dl, device, args, sacred_experiment=None, include_f1=False, mnli_subtask=None):
    model.to(device)
    model.eval()
    running_corrects, num_examples, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
    iterator = tqdm(dl, leave=False) if args.loadbar else dl

    time_start = time()

    for x1, lens, target_labels, _ in iterator:
        if task_utils.is_sentence_pair(model.cfg['task']):
            x1 = x1[0].to(device), x1[1].to(device)
            examples = len(lens[0])
        else:
            x1 = x1.to(device)
            examples = len(lens)
        target_labels = target_labels.to(device)
        torch.set_grad_enabled(False)
        out_logits = model(x1, lens)
        _, preds = torch.max(out_logits, 1)
        target_labels = target_labels.squeeze()
        running_corrects += torch.sum(preds == target_labels.data).item()
        num_examples += examples
        if include_f1:
            tp, fp, fn, tn = update_f1_counts_distilled(preds, target_labels, tp, fp, fn, tn)

    time_end = time()

    accuracy = 0 if num_examples == 0 else running_corrects / num_examples

    if include_f1:
        print(f'tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}, combined: {tp + fp + fn + tn}')
        print(f'num_examples: {num_examples}')
        f1_score = tp / (tp + 0.5 * (fp + fn))
        tn = num_examples - tp - fp - fn
        print(f'tp: {tp} ({(tp / num_examples) * 100:.2f}%)')
        print(f'fp: {fp} ({(fp / num_examples)* 100:.2f}%)')
        print(f'tn: {tn} ({(tn / num_examples) * 100:.2f}%)')
        print(f'fn: {fn} ({(fn / num_examples)*100:.2f}%)')
        print(f'Recall (Acc when label is 1): {(tp / (tp + fn))*100:.2f}%')
        print(f'Precision (Acc when predicting 1): {(tp / (tp + fp)) * 100:.2f}%')
        print(f'Acc when label is 0: {(tn / (tn + fp))*100:.2f}%')
        print(f'Acc when predicting 0: {(tn / (tn + fn))*100:.2f}%')
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("test.accuracy", accuracy)
            sacred_experiment.log_scalar("test.f1", f1_score)
        print(f'|--> eval val accuracy: {accuracy:.4f}')
        print(f'|--> eval val f1 score: {f1_score:.4f}')
    elif mnli_subtask is not None:
        if sacred_experiment is not None:
            sacred_experiment.log_scalar(f"test.{mnli_subtask}.accuracy", accuracy)
        print(f'|--> eval val {mnli_subtask} accuracy: {accuracy:.4f}')
    else:
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("test.accuracy", accuracy)
        print(f'|--> eval val accuracy: {accuracy:.4f}')

    if args.time:
        time_taken = f"{(time_end - time_start):.2f}"
        print(f'|--> eval time taken: {time_taken} seconds')

def main(args, **kwargs):
    sacred_experiment = kwargs.get("sacred_experiment")
    task = args.task
    val_data_path = task_utils.TASK_INFO[task]["path"] + '/dev.tsv'
    is_finetuned_model = model_utils.is_finetuned_model(args.arch)

    if is_finetuned_model:
        model_path = model_utils.get_model_path(args.task, "finetuned")
        model = model_utils.load_teacher(
            task, f"{model_path}/{args.model_name}", use_cpu=args.cpu
        )
        model.eval()
        if task in ['sst-2', 'rte']:
            accuracy = evaluate_accuracy(model, task, val_data_path)
            print(f'| Accuracy: {accuracy:.4f}')
            if sacred_experiment is not None:
                sacred_experiment.log_scalar("test.accuracy", accuracy)
        elif task in ['mnli']:
            for subtask in ['matched', 'mismatched']:
                val_data_path = task_utils.TASK_INFO[task]["path"] + f'/dev_{subtask}.tsv'
                print(f'{task}-{subtask}')
                accuracy = evaluate_accuracy(model, task, val_data_path)
                print(f'| Accuracy: {accuracy:.4f}')
                if sacred_experiment is not None:
                    sacred_experiment.log_scalar(f"test.{subtask}.accuracy", accuracy)
        elif task in ['qqp']:
            accuracy, f1 = evaluate_accuracy(model, task, val_data_path, include_f1=True)
            print(f'| Accuracy: {accuracy:.4f}, f1: {f1:.4f}')
            if sacred_experiment is not None:
                sacred_experiment.log_scalar("test.accuracy", accuracy)
                sacred_experiment.log_scalar("test.f1", f1)
        else:
            raise Exception(f'task {task} not currently supported')
    else: # we have a student model
        model = kwargs.get("model")

        if model is not None:
            # Check if model was quantized. Evaluate on CPU if that is the case.
            if model_utils.is_quantized_model(model):
                setattr(args, "cpu", True)

        device = torch.device('cpu') if args.cpu else torch.device('cuda')

        if model is None:
            model = distill_models.load_student(args.task, args.arch, use_gpu=not args.cpu, model_name=args.model_name)
    
        if task == 'mnli':
            for subtask in ['matched', 'mismatched']:
                val_data = data_utils.load_val_data(task, mnli_subtask=subtask)
                dl = data_utils.get_val_dataloader(model, val_data)
                evaluate_distilled_model(model, dl, device, args, sacred_experiment, mnli_subtask=subtask)
        else:
            include_f1 = task in ['qqp']
            val_data = data_utils.load_val_data(task)
            dl = data_utils.get_val_dataloader(model, val_data)
            evaluate_distilled_model(model, dl, device, args, sacred_experiment, include_f1=include_f1)
    return model

if __name__ == "__main__":
    ARGS = argparsers.args_evaluate()
    
    main(ARGS)
