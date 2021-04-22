from common.task_utils import TASK_INFO
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob

# returns sentences, labels
def load_train_data(task, ds_type='train'):
    files = [ds_type + x for x in ['.raw.input0', '.raw.input1', '.label']]
    folder = f'{TASK_INFO[task]["path"]}/processed/'

    with open(folder + files[0], encoding='utf-8') as sentences:
        with open(folder + files[2], encoding='utf-8') as targets:
            try:
                with open(folder + files[1], encoding="utf-8") as sentences2:
                    return sentences.readlines(), targets.readlines(), sentences2.readlines()
            except FileNotFoundError:
                return sentences.readlines(), targets.readlines()

# returns sentences, labels, logits
def load_distillation_data(path):
    with open(path, encoding="utf-8") as fip:
        lines = [x.strip().split("\t") for x in fip.readlines()]
        return [list(x) for x in list(zip(*lines))]

def load_val_data(task):
    return load_train_data(task, ds_type="dev")

def load_augment_data(task, augment_type):
    files = [f'{augment_type}.input0', f'{augment_type}.input1']
    folder = f'{TASK_INFO[task]["path"]}/augment_data/'

    with open(folder + files[0], encoding='utf-8') as sentences:
        try:
            with open(folder + files[1], encoding="utf-8") as sentences2:
                return sentences.readlines(), sentences2.readlines()
        except FileNotFoundError:
            return (sentences.readlines(),)

def load_all_distillation_data(task, only_original_data=False):
    base_path = f'{TASK_INFO[task]["path"]}/distillation_data'
    distillation_data = []
    if only_original_data:
        train_files = glob(f"{base_path}/train.tsv")
    else:
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
        self.lens1 = [torch.LongTensor([len(sent)]) for sent in self.sents1]
        self.lens2 = [torch.LongTensor([len(sent)]) for sent in self.sents2]
        self.labels = labels
        self.logits = logits if logits is not None else [None] * len(sents1)

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, idx):
        return self.sents1[idx], self.lens1[idx], self.sents2[idx], self.lens2[idx], self.labels[idx], self.logits[idx]

def get_datasets(model, sentences1, labels, sentences2=None, logits=None):
    label_tensors = [torch.LongTensor([model.label_dict[x.strip()]]) for x in labels]
    logit_tensors = None if logits is None else [torch.tensor([float(x) for x in xs.split(',')]) for xs in logits]
    sents1_tensors = [torch.LongTensor(model.embedding.encode(sent)) for sent in sentences1]
    if sentences2 is not None:
        sents2_tensors = [torch.LongTensor(model.embedding.encode(sent)) for sent in sentences2]
        return DistillationPairData(sents1_tensors, sents2_tensors, label_tensors, logit_tensors)
    else:
        return DistillationData(sents1_tensors, label_tensors, logit_tensors)

def get_dataloader_dict_val(model, validation_data):
    if len(validation_data) > 2:
        val_x1, val_labels, val_x2 = validation_data
        dataset = get_datasets(model, val_x1, val_labels, sentences2=val_x2)
    else:
        val_x1, val_labels = validation_data
        dataset = get_datasets(model, val_x1, val_labels)
    return DataLoader(
            dataset,
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=False,
            collate_fn=create_collate_fn(model.cfg)
        )

def get_dataloader_dict(model, distillation_data, validation_data):
    datasets = {}
    if len(distillation_data) > 3: # sentence_pairs
        train_x1, train_x2, train_labels, train_logits = distillation_data
        val_x1, val_labels, val_x2 = validation_data # this is apparently different
        datasets['train'] = get_datasets(model, train_x1, train_labels, sentences2=train_x2, logits=train_logits)
        datasets['val'] = get_datasets(model, val_x1, val_labels, sentences2=val_x2)
    else:
        train_x1, train_labels, train_logits = distillation_data
        val_x1, val_labels = validation_data
        datasets['train'] = get_datasets(model, train_x1, train_labels, logits=train_logits)
        datasets['val'] = get_datasets(model, val_x1, val_labels)
    dataloaders = {x: DataLoader(
            datasets[x],
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=x == 'train',
            collate_fn=create_collate_fn(model.cfg)) for x in ['train', 'val']}
    return dataloaders

# pads sentences in a batch to equal length
# code inspired by https://github.com/hpanwar08/sentence-classification-pytorch/
# TODO: remember to make this function, or a similar for sentence pairs
def create_collate_fn(cfg):
    pad_idx = cfg['vocab-size']
    use_hash_emb = cfg['embedding-type'] == 'hash'
    if use_hash_emb:
        num_hashes = cfg['num-hashes']
    def collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        if len(data[0]) == 4: # single sentence
            lens = [length for _,length,_,_ in data]
            labels, all_logits, lengths = [], [], []
            if use_hash_emb:
                padded_sents = torch.empty(len(data), max(lens), num_hashes).long().fill_(pad_idx)
            else:
                padded_sents = torch.empty(len(data), max(lens)).long().fill_(pad_idx)
            for i, (sent, length, label, logits) in enumerate(data):
                if use_hash_emb:
                    padded_sents[i, :lens[i],] = sent
                else:
                    padded_sents[i,:lens[i]] = sent
                labels.append(label)
                all_logits.append(logits)
                lengths.append(length)
            all_logits = torch.stack(all_logits) if all_logits[0] is not None else all_logits
            return padded_sents, torch.cat(lengths), torch.stack(labels), all_logits 
        else: # sentence pairs
            lens1 = [length for _,length,_,_,_,_ in data]
            lens2 = [length for _,_,_,length,_,_ in data]
            labels, all_logits, lengths1, lengths2 = [], [], [], []
            if use_hash_emb:
                padded_sents1 = torch.empty(len(data), max(lens1), num_hashes).long().fill_(pad_idx)
                padded_sents2 = torch.empty(len(data), max(lens2), num_hashes).long().fill_(pad_idx)
            else:
                padded_sents1 = torch.empty(len(data), max(lens1)).long().fill_(pad_idx)
                padded_sents2 = torch.empty(len(data), max(lens2)).long().fill_(pad_idx)
            for i, (sent1, length1, sent2, length2, label, logits) in enumerate(data):
                if use_hash_emb:
                    padded_sents1[i, :lens1[i],] = sent1
                    padded_sents2[i, :lens2[i],] = sent2
                else:
                    padded_sents1[i,:lens1[i]] = sent1
                    padded_sents2[i,:lens2[i]] = sent2
                labels.append(label)
                all_logits.append(logits)
                lengths1.append(length1)
                lengths2.append(length2)
            all_logits = torch.stack(all_logits) if all_logits[0] is not None else all_logits
            all_labels = torch.stack(labels)
            all_sents = (padded_sents1, padded_sents2)
            all_lengths = (torch.cat(lengths1), torch.cat(lengths2))
            return all_sents, all_lengths, all_labels, all_logits
    return collate_fn