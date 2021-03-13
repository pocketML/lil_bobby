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
    files = ['train.raw.input0', 'train.label']
    train_path = f'{TASK_INFO[task]["path"]}/processed/'
    output = []
    with open(train_path + files[0], encoding='utf-8') as fp1:
        with open(train_path + files[1], encoding='utf-8') as fp2:
            sentences, targets = fp1.readlines(), fp2.readlines()
            return sentences, targets

def generate_for_train_data(model, args):
    sentences, targets = load_train_data(args.task)
    batch_size = 32
    n = len(sentences)
    output_path = f'{TASK_INFO[args.task]["path"]}/distillation_loss/'
    with open(output_path + 'train.tsv', 'w', encoding='utf-8') as out:
        for i in range(int((n - 1) / batch_size) + 1):
            start = i * batch_size
            end = start + batch_size if (start + batch_size) < n else n
            batch_sents = sentences[start : end]
            batch_targets = targets[start : end]
            batch = collate_tokens(
                [model.encode(sent) for sent in batch_sents], 
                pad_idx=1
            )
            batch_logits = model.predict('sentence_classification_head', batch, return_logits=True)

            for sent, target, logits in zip(batch_sents, batch_targets, batch_logits.tolist()):
                logits_str = ','.join([str(x) for x in logits])
                out.write(f'{sent.strip()}\t{target.strip()}\t{logits_str}\n')


def generate(args):
    model = load_teacher(args.task, args.cpu)
    generate_for_train_data(model, args)
    