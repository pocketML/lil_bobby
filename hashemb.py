from compression.distillation.student_models import base
from embedding.hash_emb import HashEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

OUTPUT_DIM = 25

class HashEmbeddingTrainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = HashEmbedding(cfg)
        self.embedding.init_weight_range(0.1)
        self.fc1 = nn.Linear(OUTPUT_DIM, 128, bias=False)
        self.fc1.weight.data.uniform_(-0.1,0.1)
        self.fc2 = nn.Linear(128, 300, bias=False)
        self.fc2.weight.data.uniform_(-0.1,0.1)


    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
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

def save_embeddings(model):
    scalars = np.array(model.embedding.scalars.weight.cpu().data)
    vectors = np.array(model.embedding.vectors.weight.cpu().data)
    with open('data/hash_emb_pretrained.npy', 'wb') as f:
        np.save(f, scalars)
        np.save(f, vectors)

def load_embeddings(emb):
    with open('data/hash_emb_pretrained.npy', 'rb') as f:
        scalars = np.load(f)
        vectors = np.load(f)
        emb.scalars = nn.Embedding.from_pretrained(
            torch.from_numpy(scalars), 
            freeze=False
        )
        emb.vectors = nn.EmbeddingBag.from_pretrained(
            torch.from_numpy(vectors),
            mode='sum',
            freeze=False
        )

def load_glove(model):
    words, tensors = [], []
    lines = None
    with open('data/glove.42B.300d.txt', 'r', encoding='utf-8') as fip: # TODO make this download if it doesn't exist
        lines = fip.readlines()
        lines = lines[:1000000]
    for line in lines:
        line = line.strip().split()
        words.append(model.embedding.encode(line[0]))
        numbers = [float(x) for x in line[1:]]
        tensors.append(torch.FloatTensor(numbers))
    return torch.stack(words), torch.stack(tensors)

def train_on_glove(num_hashes=3, vocab_size=5000, embedding_dim=OUTPUT_DIM, hash_ratio=10, use_gpu=True):
    device =torch.device('cuda') if use_gpu else torch.device('cpu')

    cfg = base.get_default_student_config('sst-2', 'char-rnn')
    cfg['num_hashes'] = num_hashes
    cfg['vocab-size'] = vocab_size
    cfg['embedding-dim'] = embedding_dim
    cfg['hash-ratio'] = hash_ratio
    cfg['use-gpu'] = use_gpu

    model = HashEmbeddingTrainer(cfg)
    model.to(device)

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
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print("*** Preparing to train the embeddings... ***")
    best_loss = 2147483647.0
    for epoch in range(1, num_epochs + 1):
        print(f'* Epoch {epoch + 1}')
        total_loss = 0.0
        for x, y in tqdm(dl):
            x = x.to(device)
            y = y.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        print(f'|--> Loss {total_loss / len(dl.dataset):.4f}')
        if total_loss < best_loss:
            best_loss = total_loss
            print('*** Saving embeddings ***')
            save_embeddings(model)

if __name__ == "__main__":
    train_on_glove()
