from common.task_utils import TASK_INFO
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

def load_teacher(task, use_cpu):
    bin_path = f'{TASK_INFO[task]["path"]}/processed/{task}-bin/'
    model = RobertaModel.from_pretrained(
        "checkpoints", #f'models/experiments/finetune_{task}',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=bin_path
    )
    model.eval()
    if not use_cpu:
        model.cuda()
    return model

def load_train_data(task):
    files = ['train.raw.input0', 'train.raw.input1', 'train.label']
    train_path = f'{TASK_INFO[task]["path"]}/processed/'
    with open(train_path + files[0], encoding='utf-8') as sentences:
        with open(train_path + files[2], encoding='utf-8') as targets:
            try:
                with open(train_path + files[1], encoding="utf-8") as sentences2:
                    return sentences.readlines(), targets.readlines(), sentences2.readlines()
            except FileNotFoundError:
                return sentences.readlines(), targets.readlines()

def generate_for_train_data(model, args):
    data = load_train_data(args.task)
    batch_size = 32
    n = len(data[0])
    sentence_pairs = len(data) == 3
    output_path = f'{TASK_INFO[args.task]["path"]}/distillation_loss/'
    with open(output_path + 'train.tsv', 'w', encoding='utf-8') as out:
        for i in range(int((n - 1) / batch_size) + 1):
            start = i * batch_size
            end = start + batch_size if (start + batch_size) < n else n
            batch_sents = data[0][start : end]
            batch_targets = data[1][start : end]
            if sentence_pairs:
                batch_sents2 = data[2][start : end]
                batch = collate_tokens(
                    [model.encode(sent1, sent2) for sent1, sent2 in zip(batch_sents, batch_sents2)],
                    pad_idx=1
                )
            else: 
                batch = collate_tokens(
                    [model.encode(sent) for sent in batch_sents], 
                    pad_idx=1
                )
            batch_logits = model.predict('sentence_classification_head', batch, return_logits=True)

            if sentence_pairs:
                for sent1, sent2, target, logits in zip(batch_sents, batch_sents2, batch_targets, batch_logits.tolist()):
                    logits_str = ','.join([str(x) for x in logits])
                    out.write(f'{sent1.strip()}\t{sent2.strip()}\t{target.strip()}\t{logits_str}\n')
            else:
                for sent, target, logits in zip(batch_sents, batch_targets, batch_logits.tolist()):
                    logits_str = ','.join([str(x) for x in logits])
                    out.write(f'{sent.strip()}\t{target.strip()}\t{logits_str}\n')


def generate(args):
    model = load_teacher(args.task, args.cpu)
    generate_for_train_data(model, args)
    