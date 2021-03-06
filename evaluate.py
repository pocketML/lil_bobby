from fairseq.models.roberta import RobertaModel
from common import argparsers, task_utils

def sentiment_encoding(model, tokens):
    sent, target = tokens[0], tokens[1]
    encoded = model.encode(sent)
    return encoded, target

def evaluate(model, task):
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])
    ncorrect, nsamples = 0,0
    with open(task_utils.TASK_INFO[task]["path"] + '/dev.tsv') as fin:
        fin.readline()
        for i, line in enumerate(fin):
            if line.strip() == '':
                continue
            tokens = line.strip().split('\t')
            if task == 'sst-2':
                encoded, target = sentiment_encoding(model, tokens)
            else:
                sent1, sent2, target = tokens[1], tokens[2], tokens[3]
                encoded = model.encode(sent1, sent2)
            pred = model.predict('sentence_classification_head', encoded).argmax().item()
            pred_label = label_fn(pred)
            ncorrect += int(pred_label == target)
            nsamples += 1
    print(f'| Accuracy: {ncorrect/nsamples:.4f}')

def main(args):
    data_path = f'{task_utils.TASK_INFO[args.task]["path"]}/processed/{args.task}-bin/'

    model = RobertaModel.from_pretrained(
        'checkpoints',
        checkpoint_file=args.model_name,
        data_name_or_path=data_path
    )
    if not args.cpu:
        model.cuda()
    model.eval()
    evaluate(model, args.task)

if __name__ == "__main__":
    ARGS = argparsers.args_evaluate()
    main(ARGS)
