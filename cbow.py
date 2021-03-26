import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import unicodedata
from common.task_utils import TASK_INFO
from compression.distillation import data

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

# by deriving a set from raw_text, we deduplicate the array

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
        x = self.fc1(embeds)
        x = self.relu(x)
        x = self.fc2(x)
        log_probs = self.activation(x)
        return log_probs

def get_pretrained_cbow(context_size, task, embedding_dim, vocab_size, num_epochs=50):
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
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    print("Preparing to train the embeddings...")

    # prepare data for training
    context_idxs = [i for i in range(-context_size, context_size + 1) if i != 0]
    train_data = []
    for sent in tokenized_data:
        if len(sent) <= context_size * 2:
            continue
        sent_data = []
        for i in range(context_size, len(sent) - context_size):
            context = [sent[i + idx] for idx in context_idxs]
            target = sent[i]
            sent_data.append((context, target))
        train_data.append(sent_data)
    
    print(f"Sentences before prep: {len(tokenized_data)}")
    print(f'Sentences after prep:  {len(train_data)}')

    # 100 baby squats
    train_loop(train_data, model, cwobemb, criterion, num_epochs, optimizer)
    cwobemb.vectors = model.embeddings.weight
    
    return cwobemb

def train_loop(train_data, model, cwemb, criterion, num_epochs, optimizer):
    for epoch in range(num_epochs):
        print(f'* Epoch {epoch + 1}')
        total_loss = 0.0
        for sent_data in train_data:
            sent_loss = 0.0
            model.zero_grad()
            for ctx, target in sent_data:
                ctx_idxs = torch.LongTensor([cwemb.word_to_idx.get(w, cwemb.specials['unknown']) for w in ctx])

                log_probs = model(ctx_idxs)

                loss = criterion(log_probs, torch.LongTensor([cwemb.word_to_idx.get(target, cwemb.specials['unknown'])]))

                sent_loss += loss
            sent_loss.backward()
            optimizer.step()
            total_loss += sent_loss
        print(f'|--> Loss {total_loss / len(train_data):.4f}')


cwobemb = get_pretrained_cbow(2, 'sst-2', 16, 1000, 10)