import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import pickle
import numpy as np
from collections import Counter
import unicodedata

from common import data_utils
from common.model_utils import get_model_path
from embedding.base import Embedding

class CBOWDataset(Dataset):
    def __init__(self, sentences, targets):
        super().__init__()
        self.sentences = sentences
        self.targets = targets

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.targets[idx]

class CBOWEmbedding(Embedding):
    def __init__(self, cfg, load=True):
        self.vectors = None
        super().__init__(cfg, load)

    def init_embeddings(self):
        self.specials = {'unknown': 0, 'pad': 1}

        if self.load_pretrained:
            vocab = self.cfg['vocab-size']
            dim = self.cfg['embedding-dim']
            model_path = get_model_path(self.cfg["task"], "embeddings")
            self.vectors = np.load(f"{model_path}/cbow_{vocab}_{dim}_vectors.npy")
            self.vectors = torch.from_numpy(self.vectors)
            with open(f"{model_path}/cbow_{vocab}_{dim}_dict.bin", "rb") as fp:
                self.mapping = pickle.load(fp)
            embedding = nn.Embedding.from_pretrained(self.vectors).float()
        else:
            embedding = nn.Embedding(
                self.cfg['vocab-size'] + 1,
                self.cfg['embedding-dim'],
                padding_idx=self.specials['pad']
            )

        return embedding

    def tokenize(self, sentence):
        out = []
        tokenized = sentence if isinstance(sentence, list) else sentence.split()
        for word in tokenized:
            word = unicodedata.normalize('NFD', word)
            word = word.encode('ascii', 'ignore')
            word = word.decode("utf-8")
            word = word.lower()
            out.append(str(word))
        return out

    def create_vocab(self, loaded_data, vocab_size):
        tokenized_data = [self.tokenize(sent) for sent in loaded_data]
        vocab_counts = Counter([x for xs in tokenized_data for x in xs])
        vocab = list(vocab_counts.items())
        vocab.sort(key=lambda x: x[1], reverse=True)
        vocab = list(zip(*vocab[:vocab_size - 2]))[0]
        print("Vocabulary generated...")

        self.mapping = {word: i + len(self.specials) for i, word in enumerate(vocab)}

        return tokenized_data

    def encode(self, sentence):
        return torch.LongTensor([self.mapping.get(w, self.specials['unknown']) for w in self.tokenize(sentence)])

    def save(self, task):
        model_path = get_model_path(task, "embeddings")
        if self.vectors is None:
            raise ValueError("Missing trained embeddings during save!")

        vocab = self.cfg['vocab-size']
        dim = self.cfg['embedding-dim']

        np.save(f"{model_path}/cbow_{vocab}_{dim}_vectors.npy", self.vectors)
        with open(f"{model_path}/cbow_{vocab}_{dim}_dict.bin", "wb") as fp:
            pickle.dump(self.mapping, fp)

class CBOW(nn.Module):
    def __init__(self, cbowemb, vocab_size, context_size, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.embedding = cbowemb
        classifier_hidden_dim = 128
        self.fc1 = nn.Linear(cbowemb.cfg["embedding-dim"], classifier_hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(classifier_hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        x = torch.mean(embeds, dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        log_probs = self.activation(x)
        return log_probs

    def save(self, task):
        model_path = get_model_path(task, "embeddings")
        torch.save(self.state_dict(), f"{model_path}/cbow_model.pt")
        self.embedding.vectors = self.embedding.embedding.weight.detach().cpu().numpy()
        self.embedding.save(task)

    def load(self, task):
        model_path = get_model_path(task, "embeddings")
        self.load_state_dict(torch.load(f"{model_path}/cbow_model.pt"))
        self.embedding.load()
        self.eval()

def train_loop(dataloader, model, criterion, num_epochs, optimizer, task, use_cpu=False):
    for epoch in range(num_epochs):
        print(f'* Epoch {epoch + 1}')
        total_loss = 0.0
        for x, y in dataloader:
            model.zero_grad()

            if not use_cpu:
                x = x.cuda()
                y = y.cuda()

            log_probs = model(x)

            loss = criterion(log_probs, y.squeeze())

            loss.backward()
            optimizer.step()

            total_loss += loss

        print(f'|--> Loss {total_loss / len(dataloader.dataset):.4f}')

        model.save(task)

def get_dataloader(tokenized_data, cbowemb, context_size, batch_size=32):
    context_idxs = [i for i in range(-context_size, context_size + 1) if i != 0]
    train_x = []
    train_y = []
    for sent in tokenized_data:
        if len(sent) <= context_size * 2:
            continue
        for i in range(context_size, len(sent) - context_size):
            context = [sent[i + idx] for idx in context_idxs]
            target = sent[i]
            x = cbowemb.encode(context)
            y = cbowemb.encode([target])
            train_x.append(x)
            train_y.append(y)

    dataset = CBOWDataset(train_x, train_y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

def load_pretrained_embeddings(task, embedding_dim):
    cbowemb = CBOWEmbedding(None, embedding_dim=embedding_dim)
    cbowemb.load(task)

    return cbowemb

def train_embeddings(
        cbowemb, args
):
    # load data
    train_data = data_utils.load_train_data(args.task)
    loaded_data = train_data[0] if len(train_data) == 2 else train_data[0] + train_data[2]
    if not args.original_data:
        augment_data = data_utils.load_augment_data(args.task, "tinybert")
        augment_data = augment_data[0] if len(augment_data) == 1 else augment_data[0] + augment_data[1]
        loaded_data = loaded_data + augment_data
    print("Data loaded...")

    # found perfect vocab content
    tokenized_data = cbowemb.create_vocab(loaded_data, args.vocab_size)
    model = CBOW(cbowemb, args.vocab_size, args.context_size, args.batch_size)
    if not args.cpu:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    print("Preparing to train the embeddings...")

    # prepare data for training
    dataloader = get_dataloader(tokenized_data, cbowemb, args.context_size, args.batch_size)

    print(f"Sentences before prep: {len(tokenized_data)}")
    print(f'Sentences after prep:  {len(dataloader.dataset)}')

    # 100 baby squats
    train_loop(dataloader, model, criterion, args.epochs, optimizer, args.task, args.cpu)

    return cbowemb

def evaluate_cbow(cbowemb, args):
    data = data_utils.load_val_data(args.task)
    cbowemb.eval()

    encoded = cbowemb.encode(data[0])

    print(cbowemb(encoded))
