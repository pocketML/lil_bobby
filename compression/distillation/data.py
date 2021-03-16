from common.task_utils import TASK_INFO
from fairseq.data.data_utils import collate_tokens
from torch.utils.data import Dataset, DataLoader
import compression.distillation.models as models

# returns sentences, labels
def load_train_data(task, ds_type='train'):
    files = [ds_type + x for x in ['.raw.input0', '.raw.input1', '.label']]
    folder = f'{TASK_INFO[task]["path"]}/processed/'

    with open(folder + files[0], encoding='utf-8') as sentences:
        with open(folder + files[2], encoding='utf-8') as targets:
            try:
                with open(folder + files[1], encoding="utf-8") as sentences2:
                    return zip(sentences.readlines(), sentences2.readlines()), targets.readlines()
            except FileNotFoundError:
                return sentences.readlines(), targets.readlines()

def generate_for_train_data(model, args):
    data = load_train_data(args.task)
    batch_size = 32
    n = len(data[0])
    sentence_pairs = len(data) == 3
    output_path = f'{TASK_INFO[args.task]["path"]}/distillation_data/'
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


def generate_distillation_loss(args):
    model = models.load_teacher(args.task, args.cpu)
    generate_for_train_data(model, args)
    
class DistillationData(Dataset):
    def __init__(self, sents, labels, target_logits=None) -> None:
        super().__init__()
        self.sents = sents
        self.target_logits = target_logits
        self.labels = labels

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        if self.target_logits is None:
            return self.sents[index], self.labels[index]
        return self.sents[index], self.target_logits[index], self.labels[index]

def get_dataloaders(batch_size, train_x, train_y, train_logits, val_x, val_y):
    datasets = {}
    datasets['train'] = DistillationData(train_x, train_y, target_logits=train_logits)
    datasets['val'] = DistillationData(val_x, val_y)
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        ) for x in ('train', 'val')
    }
    return dataloaders

# returns sentences, labels, logits
def load_distillation_data(path):
    with open(path, encoding="utf-8") as fip:
        lines = [x.split("\t") for x in fip.readlines()]
        unzipped = list(zip(*lines))
        if len(unzipped) == 4:
            return zip(unzipped[0], unzipped[1]), unzipped[2], unzipped[3]
        else:
            return unzipped

def load_val_data(task):
    return load_train_data(task, ds_type="val")

def data_to_tensors(sentences, labels, model, logits=None):
    pass