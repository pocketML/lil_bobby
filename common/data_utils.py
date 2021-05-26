import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from glob import glob
import gc

from common.task_utils import TASK_INFO
from preprocessing import data_augment

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
def load_distillation_data(path, bootstap_data_ratio=1.0):
    with open(path, encoding="utf-8") as fip:
        to_keep = int(data_augment.N_SAMPLES * bootstap_data_ratio)
        lines = []
        for i,line in enumerate(fip):
            if i % data_augment.N_SAMPLES < to_keep and line.strip() != "":
                lines.append(line.strip().split("\t"))
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

def load_all_distillation_data(task, only_original_data=False, bootstrap_data_ratio=1.0):
    base_path = f'{TASK_INFO[task]["path"]}/distillation_data'
    train_data = []
    if only_original_data:
        train_files = glob(f"{base_path}/train.tsv")
    else:
        train_files = glob(f"{base_path}/*.tsv")
    for filename in train_files:
        if bootstrap_data_ratio < 1.0 and 'train.tsv' not in filename:
            loaded_data = load_distillation_data(filename, bootstrap_data_ratio)
        else:
            loaded_data = load_distillation_data(filename)

        if train_data == []:
            train_data = loaded_data
        else:
            train_data[0].extend(loaded_data[0])
            train_data[1].extend(loaded_data[1])
            train_data[2].extend(loaded_data[2])
            if len(train_data) > 3:
                train_data[3].extend(loaded_data[3])
    return train_data

class DistillationData(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_datasets(model, sentences1, labels, sentences2=None, logits=None, loadbar=True):
    data = []
    iterator = range(len(sentences1) - 1, -1, -1) # iterating from the back
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
    return DistillationData(data)

def get_val_dataloader(model, validation_data, loadbar=True):
    if len(validation_data) > 2:
        x1, labels, x2 = validation_data
        dataset = get_datasets(model, x1, labels, sentences2=x2, loadbar=loadbar)
    else:
        x, labels = validation_data
        dataset = get_datasets(model, x, labels, loadbar=loadbar)
    return DataLoader(
            dataset,
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=False,
            collate_fn=create_collate_fn(model.cfg)
        )

def get_train_dataloader(model, distillation_data, loadbar=True):
    if len(distillation_data) > 3:
        x1, x2, labels, logits = distillation_data
        dataset = get_datasets(model, x1, labels, sentences2=x2, logits=logits, loadbar=loadbar)
    else:
        x, labels, logits = distillation_data
        dataset = get_datasets(model, x, labels, logits=logits, loadbar=loadbar)
    return DataLoader(
            dataset,
            batch_size=model.cfg['batch-size'],
            shuffle=True,
            drop_last=True,
            collate_fn=create_collate_fn(model.cfg)
        )

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
    