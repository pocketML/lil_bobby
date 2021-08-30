"""
This module is used for evaluating the performance of various models
on SST-2 (acc), QQP (acc/f1), and MNLI (avg of matched/mismatched acc).
"""

from time import time

from tqdm import tqdm
import torch

from common import argparsers, task_utils, model_utils, data_utils
from compression.distillation import models as distill_models

# for finetuned RobertaModel
def prepare_eval_data(model, task, filepath):
    """
    Prepare evaluation data for finetuned RoBERTa model.

    Parameters
    ----------
    model : Finetuned RoBERTa Model
        The finetuned RoBERTa model to prepare data for.
    task : str
        The GLUE task to prepare data for (SST-2/QQP/MNLI).
    filepath : str
        Path to development data of given GLUE task.

    Returns
    ----------
    list[tuple[tensor, tensor]]
        List of tuples with input - target label tensors.
    """
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
    """
    Update true positive, false positive, false negative counts
    from given predictions and target labels.

    Parameters
    ----------
    pred_label : str
        Prediction provided by a given RoBERTa model.
    target : str
        Ground truth label.
    tp : int
        Running count of true positives for the current evaluation.
    fp : int
        Running count of false positives for the current evaluation.
    fn : int
        Running count of false negatives for the current evaluation.

    Returns
    ----------
    tuple[int, int, int]
        Updated count of true positives, false positives, false negatives.
    """
    tp += int(pred_label == '1' and target == '1')
    fp += int(pred_label == '1' and target == '0')
    fn += int(pred_label == '0' and target == '1')
    return tp, fp, fn

def evaluate_roberta_model(model, task, val_data_path, include_f1=False):
    """
    Evaluate accuracy of finetuned RoBERTa model on the given task.

    Parameters
    ----------
    model : Finetuned RoBERTa Model
        The finetuned RoBERTa model to evaluate performance on.
    task : str
        GLUE task to evaluate performance on.
    val_data_path : str
        Path to development data set for the given task.
    include_f1 : bool
        Whether to return F1-score in addition to accuracy (used with QQP).

    Returns
    ----------
    Union[float, tuple[float, float]]
        Average accuracy (and F1-score if included) of the model on dev set.
    """
    # Get data from GLUE dev set for given task.
    eval_data = prepare_eval_data(model, task, val_data_path)
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    ncorrect, tp, fp, fn = 0, 0, 0, 0
    # Run through each sample and use RoBERTa model to predict.
    for encoded, target in eval_data:
        pred = model.predict('sentence_classification_head', encoded).argmax().item()
        pred_label = label_fn(pred)
        ncorrect += int(pred_label == target)
        if include_f1:
            tp, fp, fn = update_f1_counts_roberta(pred_label, target, tp, fp, fn)

    accuracy = ncorrect / len(eval_data)
    if include_f1:
        # Equation for calculating F-1 score.
        f1_score = tp / (tp + 0.5 * (fp + fn))
        return (accuracy, f1_score)
    else:
        return accuracy

def update_f1_counts_distilled(pred_labels, target_labels, tp, fp, fn, tn):
    """
    Update true positive, false positive, false negative, true negative counts
    from given predictions and target labels.

    Parameters
    ----------
    pred_labels : str
        Predictions for the current batch provided by a given distilled model.
    target : str
        Ground truth labels for the current batch.
    tp : int
        Running count of true positives for the current evaluation.
    fp : int
        Running count of false positives for the current evaluation.
    fn : int
        Running count of false negatives for the current evaluation.
    tn : int
        Running count of true negatives for the current evaluation.

    Returns
    ----------
    tuple[int, int, int, int]
        Updated count of true positives, false positives, false negatives, true negatives.
    """
    pred_ones = pred_labels == 1
    target_ones = target_labels == 1
    pred_zeros = pred_labels == 0
    target_zeros = target_labels == 0
    tp += torch.count_nonzero(torch.logical_and(pred_ones, target_ones)).item()
    fp += torch.count_nonzero(torch.logical_and(pred_ones, target_zeros)).item()
    fn += torch.count_nonzero(torch.logical_and(pred_zeros, target_ones)).item()
    tn += torch.count_nonzero(torch.logical_and(pred_zeros, target_zeros)).item()
    return tp, fp, fn, tn

def evaluate_distilled_model(model, dl, device, args, sacred_experiment=None, include_f1=False, mnli_subtask=None, include_time=False):
    """
    Evaluate accuracy of distilled model on the given task.

    Parameters
    ----------
    model : Distilled student model
        The distilled model being evaluated on.
    dl : torch.data.utils.Dataloader
        Dataloader containing data from development set for a GLUE task.
    device : torch.device
        Device on which to run the evaluation.
    args : Namespace
        Additional args.
    sacred_experiment : Run
        Sacred experiment instance to save evaluation data to.
    include_f1 : bool
        Whether to return F1-score in addition to accuracy (used with QQP).
    mnli_subtask : str
        Which MNLI subtask to run evaluation for, when task=MNLI. Either 'matched'/'mismatched'.
    include_time : bool
        Whether to record how long the evaluation took.
    """
    # Set model up for evaluation.
    model.to(device)
    model.eval()
    running_corrects, num_examples, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
    iterator = tqdm(dl, leave=False) if args.loadbar else dl

    time_start = time()

    # Do the evaluation in batches.
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

    # Print all the stuff and log to Sacred if an experiment is given.
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

    if include_time:
        time_taken = f"{(time_end - time_start):.2f}"
        print(f'|--> eval time taken: {time_taken} seconds')

def main(args, **kwargs):
    # Get Sacred experiment (if it exists) and other args.
    sacred_experiment = kwargs.get("sacred_experiment")
    task = args.task
    val_data_path = task_utils.TASK_INFO[task]["path"] + '/dev.tsv'
    is_finetuned_model = model_utils.is_finetuned_model(args.arch)

    if is_finetuned_model: # Evaluate finetuned RoBERTa model.
        model_path = model_utils.get_model_path(args.task, "finetuned")
        model = model_utils.load_teacher(
            task, f"{model_path}/{args.model_name}", use_cpu=args.cpu
        )
        model.eval()
        if task in ['sst-2', 'rte']:
            accuracy = evaluate_roberta_model(model, task, val_data_path)
            print(f'| Accuracy: {accuracy:.4f}')
            if sacred_experiment is not None:
                sacred_experiment.log_scalar("test.accuracy", accuracy)
        elif task in ['mnli']:
            # Evaluate on each MNLI subtask in sequence.
            for subtask in ['matched', 'mismatched']:
                val_data_path = task_utils.TASK_INFO[task]["path"] + f'/dev_{subtask}.tsv'
                print(f'{task}-{subtask}')
                accuracy = evaluate_roberta_model(model, task, val_data_path)
                print(f'| Accuracy: {accuracy:.4f}')
                if sacred_experiment is not None:
                    sacred_experiment.log_scalar(f"test.{subtask}.accuracy", accuracy)
        elif task in ['qqp']:
            # Evaluate both accuracy and F1-score for QQP.
            accuracy, f1 = evaluate_roberta_model(model, task, val_data_path, include_f1=True)
            print(f'| Accuracy: {accuracy:.4f}, f1: {f1:.4f}')
            if sacred_experiment is not None:
                sacred_experiment.log_scalar("test.accuracy", accuracy)
                sacred_experiment.log_scalar("test.f1", f1)
        else:
            raise Exception(f'task {task} not currently supported')
    else: # Evaluate distilled student model.
        model = kwargs.get("model")

        if model is not None:
            # Check if model was quantized. Evaluate on CPU if that is the case.
            if model_utils.is_quantized_model(model):
                setattr(args, "cpu", True)

        device = torch.device('cpu') if args.cpu else torch.device('cuda')

        if model is None:
            model = distill_models.load_student(args.task, args.arch, use_gpu=not args.cpu, model_name=args.model_name)
    
        if task == 'mnli':
            # Evaluate for each MNLI subtask.
            for subtask in ['matched', 'mismatched']:
                val_data = data_utils.load_val_data(task, mnli_subtask=subtask)
                dl = data_utils.get_val_dataloader(model, val_data)
                evaluate_distilled_model(model, dl, device, args, sacred_experiment, mnli_subtask=subtask, include_time=args.time)
        else: # Include F-1 if task is QQP.
            include_f1 = task in ['qqp']
            val_data = data_utils.load_val_data(task)
            dl = data_utils.get_val_dataloader(model, val_data)
            evaluate_distilled_model(model, dl, device, args, sacred_experiment, include_f1=include_f1, include_time=args.time)
    return model

if __name__ == "__main__":
    ARGS = argparsers.args_evaluate()
    
    main(ARGS)
