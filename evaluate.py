from fairseq.models.roberta import RobertaModel
from common import argparsers, task_utils

def prepare_eval_data(model, task, filepath):
    eval_data = []
    with open(filepath, encoding="utf8") as fin:
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

def update_f1_counts(pred, target, tp, fp, fn):
    tp += int(pred_label == '1' and target == '1')
    fp += int(pred_label == '1' and target == '0')
    fn += int(pred_label == '0' and target == '1')
    return tp, fp, fn

# f1 is the harmonic mean of the precision and recall
def evaluate_accuracy(model, task, val_data_path, include_f1=False):
    eval_data = prepare_eval_data(model, task, val_data_path)
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    ncorrect, tp, fp, fn = 0, 0, 0, 0
    for encoded, target in eval_data:
        pred = model.predict('sentence_classification_head', encoded).argmax().item()
        pred_label = label_fn(pred)
        ncorrect += int(pred_label == target)
        if include_f1:
            tp, fp, fn = update_f1_counts(pred_label, target, tp, fp, fn)

    accuracy = ncorrect / len(eval_data)
    if include_f1:
        f1_score = tp / (tp + 0.5 * (fp + fn))
        return (accuracy, f1_score)
    else:
        return accuracy

def main(args, sacred_experiment=None):
    data_path = f'{task_utils.TASK_INFO[args.task]["path"]}/processed/{args.task}-bin/'

    model = RobertaModel.from_pretrained(
        'checkpoints', # TODO should be fixed eventually
        checkpoint_file=args.model_name,
        data_name_or_path=data_path
    )
    if not args.cpu:
        model.cuda()
    model.eval()

    task = args.task
    val_data_path = task_utils.TASK_INFO[task]["path"] + '/dev.tsv'

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
                sacred_experiment.log_scalar(f"test.accuracy.{subtask}", accuracy)
    elif task in ['qqp']:
        accuracy, f1 = evaluate_accuracy(model, task, val_data_path, include_f1=True)
        print(f'| Accuracy: {accuracy:.4f}, f1: {f1:.4f}')
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("test.accuracy", accuracy)
            sacred_experiment.log_scalar("test.f1", f1)
    else:
        raise Exception(f'task {task} not currently supported')

if __name__ == "__main__":
    ARGS = argparsers.args_evaluate()
    
    main(ARGS)
