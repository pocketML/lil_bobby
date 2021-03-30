import torch
from common import argparsers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import unicodedata
from compression.distillation import data
from common.task_utils import get_model_path

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
        self.specials = None

    def tokenize(self, sentence):
        out = []
        for word in sentence.split():
            word = unicodedata.normalize('NFD', word)
            word = word.encode('ascii', 'ignore')
            word = word.decode("utf-8")
            word = word.lower()
            out.append(str(word))
        return out

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, specials):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=specials['pad'])
        self.fc1 = nn.Linear(context_size * embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        print(embeds.shape)
        x = self.fc1(embeds)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.fc2(x)
        log_probs = self.activation(x)
        return log_probs

    def save(self, task):
        model_path = get_model_path(task, "embeddings")
        torch.save(self.state_dict(), f"{model_path}/cbow.pt")

    def load(self, task):
        model_path = get_model_path(task, "embeddings")
        self.load_state_dict(torch.load(f"{model_path}/cbow.pt"))
        self.eval()

def train_loop(dataloader, model, cwemb, criterion, num_epochs, optimizer, use_cpu=False):
    for epoch in range(num_epochs):
        print(f'* Epoch {epoch + 1}')
        total_loss = 0.0
        for x, y in dataloader:
            model.zero_grad()
            ctx_idxs = torch.LongTensor([cwemb.word_to_idx.get(w, cwemb.specials['unknown']) for w in y])
            target = torch.LongTensor([cwemb.word_to_idx.get(y, cwemb.specials['unknown'])])
            if not use_cpu:
                ctx_idxs = ctx_idxs.cuda()
                target = target.cuda()

            log_probs = model(ctx_idxs)

            loss = criterion(log_probs, target)

            loss.backward()
            optimizer.step()

            total_loss += loss
        print(f'|--> Loss {total_loss / len(dataloader.dataset):.4f}')

def get_dataloader(tokenized_data, context_size, batch_size=32):
    context_idxs = [i for i in range(-context_size, context_size + 1) if i != 0]
    train_x = []
    train_y = []
    for sent in tokenized_data:
        if len(sent) <= context_size * 2:
            continue
        for i in range(context_size, len(sent) - context_size):
            context = [sent[i + idx] for idx in context_idxs]
            target = sent[i]
            train_x.append(context)
            train_y.append(target)

    dataset = CBOWDataset(train_x, train_y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

def get_pretrained_cbow(context_size, task, embedding_dim, vocab_size, num_epochs=50, use_cpu=False):
    # load data
    loaded_data = data.load_all_distillation_data(task)
    loaded_data = loaded_data[0] if len(loaded_data) == 3 else loaded_data[0] + loaded_data[1]
    print("Data loaded...")

    # found perfect vocab content
    cwobemb = CBOWEmbeddings(embedding_dim)
    tokenized_data = [cwobemb.tokenize(sent) for sent in loaded_data]
    vocab_counts = Counter([x for xs in tokenized_data for x in xs])
    vocab = list(vocab_counts.items())
    vocab.sort(key=lambda x: x[1], reverse=True)
    vocab = list(zip(*vocab[:vocab_size - 2]))[0]
    print("Vocabulary generated...")

    # embeddings and model stuff
    cwobemb.specials = {'unknown': 0, 'pad': 1}
    cwobemb.word_to_idx = {word: i + len(cwobemb.specials) for i, word in enumerate(vocab)}
    model = CBOW(vocab_size, embedding_dim, context_size, cwobemb.specials)
    if not use_cpu:
        model = model.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    print("Preparing to train the embeddings...")

    # prepare data for training
    dataloader = get_dataloader(tokenized_data, context_size)

    print(f"Sentences before prep: {len(tokenized_data)}")
    print(f'Sentences after prep:  {len(dataloader.dataset)}')

    # 100 baby squats
    try:
        train_loop(dataloader, model, cwobemb, criterion, num_epochs, optimizer)
    finally:
        cwobemb.vectors = model.embeddings.weight
        model.save(task)
    
    return cwobemb

def main(args, sacred_experiment=None):
    cbowemb = get_pretrained_cbow(
        args.context_size, args.task, args.embed_dim,
        args.vocab_size, args.epochs, args.cpu
    )

if __name__ == "__main__":
    ARGS = argparsers.args_cbow()
    main(ARGS)
