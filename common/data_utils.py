import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import random
from glob import glob
import gc

from common.task_utils import TASK_INFO

class DataSplitter(list):
    def __init__(self, chunk_ratio=1.0, data_ratio=1.0):
        super().__init__()
        self.chunk_ratio = chunk_ratio
        self.data_ratio = data_ratio
        self.chunk_size = None

    def shuffle(self):
        random.shuffle(self)

    def set_chunk_size(self):
        self.chunk_size = int(len(self) * self.chunk_ratio)
        self.shuffle()

# returns sentences, labels
def load_train_data(task, ds_type='train'):
    folder = f'{TASK_INFO[task]["path"]}/processed/'
    files = [ds_type + x for x in ['.raw.input0', '.raw.input1', '.label']]
    with open(folder + files[0], encoding='utf-8') as sentences:
        with open(folder + files[2], encoding='utf-8') as targets:
            try:
                with open(folder + files[1], encoding="utf-8") as sentences2:
                    return sentences.readlines(), targets.readlines(), sentences2.readlines()
            except FileNotFoundError:
                return sentences.readlines(), targets.readlines()

# returns sentences, labels, logits
def load_distillation_data(path, data_splitter=None):
    if data_splitter is not None and data_splitter == []:
            
        skip_sentences = int(1 / data_splitter.data_ratio)

        with open(path, encoding="utf-8") as fip:
            index = 0
            for line in fip:
                line = line.strip()
                if line != "" and index % skip_sentences == 0:
                    data_splitter.append(index)
                index += 1

        # Repopulate data_splitter with index of all sentences
        data_splitter.set_chunk_size()

    with open(path, encoding="utf-8") as fip:
        lines = []
        sentences_to_include = set()
        if data_splitter is not None:
            for _ in range(data_splitter.chunk_size):
                sentences_to_include.add(data_splitter.pop())
            if len(data_splitter) < 256:
                data_splitter.clear()

        i = 0
        for line in fip:
            line = line.strip()
            if line != "" and (data_splitter is None or i in sentences_to_include):
                line = line.strip().split("\t")
                lines.append(line)
            i += 1
        return [list(x) for x in list(zip(*lines))]

def load_val_data(task, mnli_subtask='both'):
    if task == 'mnli':
        if mnli_subtask == 'both':
            matched = load_train_data(task, ds_type='dev_matched')
            mismatched = load_train_data(task, ds_type='dev_mismatched')
            return matched[0] + mismatched[0], matched[1] + mismatched[1], matched[2] + mismatched[2]
        else: # should be either matched or mismatched:
            return load_train_data(task, ds_type='dev_' + mnli_subtask)
    else:
        return load_train_data(task, ds_type='dev')

def load_augment_data(task, augment_type):
    files = [f'{augment_type}.input0', f'{augment_type}.input1']
    folder = f'{TASK_INFO[task]["path"]}/augment_data/'

    with open(folder + files[0], encoding='utf-8') as sentences:
        try:
            with open(folder + files[1], encoding="utf-8") as sentences2:
                return sentences.readlines(), sentences2.readlines()
        except FileNotFoundError:
            return (sentences.readlines(),)

def load_all_distillation_data(task, only_original_data=False, data_splitter=None):
    base_path = f'{TASK_INFO[task]["path"]}/distillation_data'
    distillation_data = []
    if only_original_data:
        train_files = glob(f"{base_path}/train.tsv")
    else:
        train_files = glob(f"{base_path}/*.tsv")
    for filename in train_files:
        if task in ['qqp', 'mnli'] and 'train.tsv' not in filename: # TODO: this logic should reside in augment and not here
            loaded_data = load_distillation_data(filename, data_splitter)
        else:
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
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DistillationPairData(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_datasets(model, sentences1, labels, sentences2=None, logits=None, loadbar=True):
    # labels
    data = []
    iterator = range(len(sentences1) - 1, -1, -1)
    if loadbar:
        iterator = tqdm(range(len(sentences1) - 1, -1, -1), leave=False)

    for i in iterator:
        # label
        label_tensor = torch.LongTensor([model.label_dict[labels[i].strip()]])
        del labels[i]

        # logit
        if logits is None:
            logit_tensor = None
        else:
            logit_tensor = torch.tensor([float(x) for x in logits[i].split(',')])
            del logits[i]
        # sentences 1 encoding and len
        encoded = model.embedding.encode(sentences1[i])
        len1 = torch.LongTensor([len(encoded)])
        sents1_tensor = torch.LongTensor(encoded)
        del sentences1[i]
        
        # sentences2 to len and encoded tensor
        if sentences2 is not None:
            encoded = model.embedding.encode(sentences2[i])
            len2 = torch.LongTensor([len(encoded)])
            sents2_tensor = torch.LongTensor(encoded)
            del sentences2[i]
            data.append((sents1_tensor, len1, sents2_tensor,  len2, label_tensor, logit_tensor))
        else:
            data.append((sents1_tensor, len1, label_tensor, logit_tensor))
        
    sentences1.clear()
    del sentences1
    labels.clear()
    del labels
    if logits is not None:
        logits.clear()
        del logits
    
    if sentences2 is not None:
        sentences2.clear()
        del sentences2
        gc.collect()
        return DistillationPairData(data)
    else:
        gc.collect()
        return DistillationData(data)

def get_dataloader_dict_val(model, validation_data, loadbar=True):
    if len(validation_data) > 2:
        val_x1, val_labels, val_x2 = validation_data
        dataset = get_datasets(model, val_x1, val_labels, sentences2=val_x2, loadbar=loadbar)
    else:
        val_x1, val_labels = validation_data
        dataset = get_datasets(model, val_x1, val_labels, loadbar=loadbar)
    return DataLoader(
            dataset,
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=False,
            collate_fn=create_collate_fn(model.cfg)
        )

def get_dataload_dict_train(model, distillation_data, loadbar=True):
    if len(distillation_data) > 3:
        train_x1, train_x2, train_labels, train_logits = distillation_data
        dataset = get_datasets(model, train_x1, train_labels, sentences2=train_x2, logits=train_logits, loadbar=loadbar)
    else:
        train_x1, train_labels, train_logits = distillation_data
        dataset = get_datasets(model, train_x1, train_labels, logits=train_logits, loadbar=loadbar)
    return DataLoader(
            dataset,
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=True,
            collate_fn=create_collate_fn(model.cfg)
        )

def get_dataloader_dict(model, distillation_data, validation_data):
    datasets = {
        "train": get_dataload_dict_train(model, distillation_data),
        "val": get_dataloader_dict_val(model, validation_data)
    }

    dataloaders = {x: DataLoader(
            datasets[x],
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=x == 'train',
            collate_fn=create_collate_fn(model.cfg)) for x in ("train", "val")}
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
                    padded_sents[i, :lens[i]] = sent
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
    