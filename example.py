from compression.distillation.student_models import base
from embedding.hash_emb import HashEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = HashEmbedding(cfg)
        self.fc = nn.Linear(100,300)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class GloveDataset(Dataset):
    def __init__(self, train_data):
        super().__init__()
        self.words = train_data[0]
        self.logits = train_data[1]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx], self.logits[idx]

def load_glove(model):
    words, tensors = [], []
    with open('data/glove.42B.300d.txt', 'r', encoding='utf-8') as fip:
        lines = fip.readlines()
        for line in lines[:1000000]:
            line = line.strip().split()
            words.append(model.embedding.encode(line[0]))
            numbers = [float(x) for x in line[1:]]
            tensors.append(torch.FloatTensor(numbers))
    return torch.stack(words), torch.stack(tensors)

def train_on_glove(num_hashes=3, vocab_size=5000, embedding_dim=100, hash_ratio=10, use_gpu=False):
    cfg = base.get_default_student_config('sst-2', 'char-rnn')
    cfg['num_hashes'] = num_hashes
    cfg['vocab-size'] = vocab_size
    cfg['embedding-dim'] = embedding_dim
    cfg['hash-ratio'] = hash_ratio
    cfg['use-gpu'] = use_gpu
    model = Model(cfg)

    train_data = load_glove(model)
    dataset = GloveDataset(train_data)
    batch_size = 100
    num_epochs = 100

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    print("*** Data loaded... ***")

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    print("*** Preparing to train the embeddings... ***")

    for epoch in range(1, num_epochs + 1):
        print(f'* Epoch {epoch + 1}')
        total_loss = 0.0
        for x, y in tqdm(dl):
            model.zero_grad()

            log_probs = model(x)
            loss = criterion(log_probs, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss * len(x)
        print(f'|--> Loss {total_loss / len(dl.dataset):.4f}')

train_on_glove()

