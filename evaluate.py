from fairseq.models.roberta import RobertaModel
from common import argparsers, task_utils

def prepare_eval_data(model, task, data_path):
    eval_data = []
    with open(data_path, encoding="utf-8") as fin:
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
            eval_data.append((encoded, target))
    return eval_data

def evaluate_accuracy(model, task):
    data_path = task_utils.TASK_INFO[task]["path"] + '/dev.tsv'
    eval_data = prepare_eval_data(model, task, data_path)
    ncorrect = 0
    nsamples = len(eval_data)
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    for encoded, target in eval_data:
        pred = model.predict('sentence_classification_head', encoded).argmax().item()
        pred_label = label_fn(pred)
        ncorrect += int(pred_label == target)
    return ncorrect/nsamples

# f1 is the harmonic mean of the precision and recall
def evaluate_accuracy_f1(model, task):
    data_path = task_utils.TASK_INFO[task]["path"] + '/dev.tsv'
    eval_data = prepare_eval_data(model, task, data_path)
    ncorrect = 0
    nsamples = len(eval_data)
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    # f1 relevant values
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for encoded, target in eval_data:
        pred = model.predict('sentence_classification_head', encoded).argmax().item()
        pred_label = label_fn(pred)
        ncorrect += int(pred_label == target)
        true_positives += int(pred_label == '1' and target == '1')
        false_positives += int(pred_label == '1' and target == '0')
        false_negatives += int(pred_label == '0' and target == '1')
    accuracy = ncorrect / nsamples
    f1_score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
    return (accuracy, f1_score)

def main(args, sacred_experiment=None):
    data_path = f'{task_utils.TASK_INFO[args.task]["path"]}/processed/{args.task}-bin/'

    model = RobertaModel.from_pretrained(
        'checkpoints',
        checkpoint_file=args.model_name,
        data_name_or_path=data_path
    )
    if not args.cpu:
        model.cuda()
    model.eval()

    if args.task in ['sst-2', 'rte', 'mnli']:
        accuracy = evaluate_accuracy(model, args.task)
        print(f'| Accuracy: {accuracy:.4f}')
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("test.accuracy", accuracy)
    elif args.task in ['qqp']:
        accuracy, f1 = evaluate_accuracy_f1(model, args.task)
        print(f'| Accuracy: {accuracy:.4f}, f1: {f1:.4f}')
        if sacred_experiment is not None:
            sacred_experiment.log_scalar("test.accuracy", accuracy)
            sacred_experiment.log_scalar("test.f1", f1)
    else:
        raise Exception(f'task {args.task} not currently supported')

if __name__ == "__main__":
    ARGS = argparsers.args_evaluate()
    
    main(ARGS)
