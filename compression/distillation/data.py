from common.task_utils import TASK_INFO
from fairseq.data.data_utils import collate_tokens
from torch.utils.data import Dataset, DataLoader
import torch
import compression.distillation.models as models
import os
from glob import glob

# returns sentences, labels
def load_train_data(task, ds_type='train', data_folder="processed"):
    files = [ds_type + x for x in ['.raw.input0', '.raw.input1', '.label']]
    folder = f'{TASK_INFO[task]["path"]}/{data_folder}/'

    with open(folder + files[0], encoding='utf-8') as sentences:
        with open(folder + files[2], encoding='utf-8') as targets:
            try:
                with open(folder + files[1], encoding="utf-8") as sentences2:
                    return sentences.readlines(), targets.readlines(), sentences2.readlines()
            except FileNotFoundError:
                return sentences.readlines(), targets.readlines()

def load_all_distillation_data(task):
    base_path = f'{TASK_INFO[task]["path"]}/distillation_data'
    distillation_data = []
    train_files = glob(f"{base_path}/*.tsv")
    for filename in train_files:
        loaded_data = load_distillation_data(filename)

        if distillation_data == []:
            distillation_data = loaded_data
        else:
            distillation_data[0].extend(loaded_data[0])
            distillation_data[1].extend(loaded_data[1])
            distillation_data[2].extend(loaded_data[2])
            if len(distillation_data) > 3:
                distillation_data[3].extend(loaded_data[3])
    return distillation_data

def generate_for_train_data(model, args):
    default_data = args.generate_loss == "default"
    input_folder = "processed" if default_data else "augment_data"
    data = load_train_data(args.task, data_folder=input_folder)
    batch_size = 8
    n = len(data[0])
    sentence_pairs = len(data) == 3
    output_path = f'{TASK_INFO[args.task]["path"]}/distillation_data/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_file = "train.tsv" if default_data else "augmented.tsv"

    with open(output_path + output_file, 'w', encoding='utf-8') as out:
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
    def __init__(self, sents, labels, logits=None) -> None:
        super().__init__()
        self.sents = sents
        self.lengths = [torch.LongTensor([len(sent)]) for sent in self.sents]
        self.labels = labels
        self.logits = logits if logits is not None else [None] * len(sents)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx], self.lengths[idx], self.labels[idx], self.logits[idx]

class DistillationPairData(Dataset):
    def __init__(self, sents1, sents2, labels, logits=None) -> None:
        super().__init__()
        self.sents1 = sents1
        self.sents2 = sents2
        self.lens1 = [torch.LongTensor(len(sent)) for sent in self.sents1]
        self.lens2 = [torch.LongTensor(len(sent)) for sent in self.sents2]
        self.labels = labels
        self.logits = logits if logits is not None else [None] * len(sents1)

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, idx):
        return self.sents1[idx], self.lens1[idx], self.sents2[idx], self.lens2[idx], self.labels[idx], self.logits[idx]

def get_datasets(model, sentences1, labels, sentences2=None, logits=None):
    label_tensors = [torch.LongTensor([model.label_dict[x.strip()]]) for x in labels]
    logit_tensors = None if logits is None else [torch.tensor([float(x) for x in xs.split(',')]) for xs in logits]
    sents1_tensors = [torch.LongTensor(model.bpe.encode_ids(sent)) for sent in sentences1]
    if sentences2 is not None:
        sents2_tensors = [torch.LongTensor(model.bpe.encode_ids(sent)) for sent in sentences2]
        return DistillationPairData(sents1_tensors, sents2_tensors, label_tensors, logit_tensors)
    else:
        return DistillationData(sents1_tensors, label_tensors, logit_tensors)

def get_dataloader_dict(model, distillation_data, validation_data):
    datasets = {}
    if len(distillation_data) > 3:
        train_x1, train_x2, train_labels, train_logits = distillation_data
        val_x1, val_x2, val_labels = validation_data
        datasets['train'] = get_datasets(model, train_x1, train_labels, sentences2=train_x2, logits=train_logits)
        datasets['val'] = get_datasets(model, val_x1, val_labels, sentences2=val_x2)
    else:
        train_x1, train_labels, train_logits = distillation_data
        val_x1, val_labels = validation_data
        datasets['train'] = get_datasets(model, train_x1, train_labels, logits=train_logits)
        datasets['val'] = get_datasets(model, val_x1, val_labels)
    dataloaders = {x: DataLoader(
            datasets[x],
            batch_size=model.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=create_collate_fn(model.cfg.vocab_size)) for x in ['train', 'val']}
    return dataloaders

# pads sentences in a batch to equal length
# code inspired by https://github.com/hpanwar08/sentence-classification-pytorch/
# TODO: remember to make this function, or a similar for sentence pairs
def create_collate_fn(pad_idx):
    def collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        if len(data[0]) == 4: # single sentence
            lens = [length for _,length,_,_ in data]
            labels, all_logits, lengths = [], [], []
            padded_sents = torch.empty(len(data), max(lens)).long().fill_(pad_idx)
            for i, (sent, length, label, logits) in enumerate(data):
                padded_sents[i,:lens[i]] = sent
                labels.append(label)
                all_logits.append(logits)
                lengths.append(length)
            all_logits = torch.stack(all_logits) if all_logits[0] is not None else all_logits
            return padded_sents, torch.cat(lengths), torch.stack(labels), all_logits 
        else:
            raise Exception("please don't be here")
    return collate_fn

# returns sentences, labels, logits
def load_distillation_data(path):
    with open(path, encoding="utf-8") as fip:
        lines = [x.strip().split("\t") for x in fip.readlines()]
        return [list(x) for x in list(zip(*lines))]

def load_val_data(task):
    return load_train_data(task, ds_type="dev")