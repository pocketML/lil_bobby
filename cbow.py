import pickle
import torch
import numpy as np
from common import argparsers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import unicodedata
from common import data_utils
from common.model_utils import get_model_path

class CBOWDataset(Dataset):
    def __init__(self, sentences, targets):
        super().__init__()
        self.sentences = sentences
        self.targets = targets

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.targets[idx]

class CBOWEmbeddings():
    def __init__(self, embedding_dim):
        self.vectors = None
        self.word_to_idx = None
        self.embedding_dim = embedding_dim
        self.specials = {'unknown': 0, 'pad': 1}

    def tokenize(self, sentence):
        out = []
        for word in sentence.split():
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

        self.word_to_idx = {word: i + len(self.specials) for i, word in enumerate(vocab)}

        return tokenized_data

    def encode(self, sentence):
        return torch.LongTensor([self.word_to_idx.get(w, self.specials['unknown']) for w in sentence])

    def save(self, task):
        model_path = get_model_path(task, "embeddings")
        if self.vectors is None:
            raise ValueError("Missing trained embeddings during save!")
        np.save(f"{model_path}/vectors.npy", self.vectors)
        with open(f"{model_path}/dict.bin", "wb") as fp:
            pickle.dump(self.word_to_idx, fp)

    def load(self, task):
        model_path = get_model_path(task, "embeddings")
        self.vectors = np.load(f"{model_path}/vectors.npy")
        with open(f"{model_path}/dict.bin", "rb") as fp:
            self.word_to_idx = pickle.load(fp)

class CBOW(nn.Module):
    def __init__(self, cbowemb, vocab_size, context_size, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.cbowemb = cbowemb
        self.embeddings = nn.Embedding(vocab_size, cbowemb.embedding_dim, padding_idx=cbowemb.specials['pad'])
        self.fc1 = nn.Linear(context_size * cbowemb.embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(self.batch_size, -1)
        x = self.fc1(embeds)
        x = self.relu(x)
        x = self.fc2(x)
        log_probs = self.activation(x)
        return log_probs

    def save(self, task):
        model_path = get_model_path(task, "embeddings")
        torch.save(self.state_dict(), f"{model_path}/cbow.pt")
        self.cbowemb.vectors = self.embeddings.weight.detach().numpy()
        self.cbowemb.save(task)

    def load(self, task):
        model_path = get_model_path(task, "embeddings")
        self.load_state_dict(torch.load(f"{model_path}/cbow.pt"))
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
    cbowemb = CBOWEmbeddings(embedding_dim)
    cbowemb.load(task)

    return cbowemb

def train_embeddings(
        context_size, task, embedding_dim,
        vocab_size, num_epochs=50,
        batch_size=32, use_cpu=False
):
    # load data
    train_data = data_utils.load_train_data(task)
    train_data = train_data[0] if len(train_data) == 2 else train_data[0] + train_data[2]
    augment_data = data_utils.load_augment_data(task, "tinybert")
    augment_data = augment_data[0] if len(augment_data) == 1 else augment_data[0] + augment_data[1]
    loaded_data = train_data + augment_data
    print("Data loaded...")

    # found perfect vocab content
    cbowemb = CBOWEmbeddings(embedding_dim)
    tokenized_data = cbowemb.create_vocab(loaded_data, vocab_size)
    model = CBOW(cbowemb, vocab_size, context_size, batch_size)
    if not use_cpu:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    print("Preparing to train the embeddings...")

    # prepare data for training
    dataloader = get_dataloader(tokenized_data, cbowemb, context_size, batch_size)

    print(f"Sentences before prep: {len(tokenized_data)}")
    print(f'Sentences after prep:  {len(dataloader.dataset)}')

    # 100 baby squats
    train_loop(dataloader, model, criterion, num_epochs, optimizer, task, use_cpu)

    return cbowemb

def main(args, sacred_experiment=None):
    # cbowemb = train_embeddings(
    #     args.context_size, args.task,
    #     args.embed_dim, args.vocab_size,
    #     args.epochs, args.batch_size, args.cpu
    # )
    load_pretrained_embeddings("sst-2", 16)

if __name__ == "__main__":
    ARGS = argparsers.args_cbow()
    main(ARGS)
