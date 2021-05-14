import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.quantized as quantized
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import hashlib
import numpy as np
from tqdm import tqdm

from embedding.base import Embedding
from compression.distillation.student_models import base

from compression.distillation.student_models import base
from preprocessing import download

class HashEmbedding(Embedding):
    def __init__(self, cfg, load=False):
        super().__init__(cfg, load)
        if self.load_pretrained:
            self.load_pretrained_on_glove()
            self.set_freeze_status()

    def init_embeddings(self):
        self.num_hashes = self.cfg['num-hashes']
        self.embedding_dim = self.cfg['embedding-dim']
        self.K = self.cfg['vocab-size']
        self.ratio = self.cfg['hash-ratio']
        self.B = self.K // self.ratio
        self.vocab_size = self.K
        scalar_size = self.vocab_size * self.num_hashes + self.num_hashes
        self.scalars = nn.Embedding(scalar_size, 1)
        self.vectors = nn.EmbeddingBag(self.B + 1, self.embedding_dim, mode='sum')
        self.hash_offsets = torch.LongTensor([i * (self.K + 1) for i in range(self.num_hashes)])
        return None

    def set_freeze_status(self):
        self.scalars.weight.requires_grad = not self.cfg["embedding-freeze"]
        self.vectors.weight.requires_grad = not self.cfg["embedding-freeze"]

    def encode(self, sent):
        sent_stack = []
        for word in sent.split(" "):
            word_stack = []
            for i in range(self.num_hashes):
                salted_word =  f'{i}{word}'
                hashed = hashlib.md5(salted_word.encode('utf-8')).digest()[-4:]
                hashed_int = int.from_bytes(hashed, 'big') % self.K
                word_stack.append(hashed_int)
            sent_stack.append(torch.LongTensor(word_stack))
        out = torch.stack(sent_stack)
        return out

    def load_pretrained_on_glove(self):
        fname = f'{BASE_FOLDER}/hash_emb_pretrained_{self.cfg["embedding-dim"]}_dim.npy'
        with open(fname, 'rb') as f:
            scalars = np.load(f)
            vectors = np.load(f)
            self.scalars = nn.Embedding.from_pretrained(torch.from_numpy(scalars))
            self.vectors = nn.EmbeddingBag.from_pretrained(torch.from_numpy(vectors), mode='sum')

    def save_embeddings(self):
        scalars = np.array(self.embedding.scalars.weight.cpu().data)
        vectors = np.array(self.embedding.vectors.weight.cpu().data)
        os.makedirs(BASE_FOLDER, exist_ok=True)
        fname = f'{BASE_FOLDER}/hash_emb_pretrained_{self.cfg["embedding-dim"]}_dim.npy'
        with open(fname, 'wb') as f:
            np.save(f, scalars)
            np.save(f, vectors)

    # is inplace
    def prepare_to_quantize(self):
        self.vectors.qconfig = quant.float_qparams_weight_only_qconfig
        self.vectors = quantized.EmbeddingBag.from_float(self.vectors)
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        quant.prepare(self.vectors, inplace=True)

    # is inplace
    def convert_to_quantized(self):
        quant.convert(self.vectors, inplace=True)

    def _apply(self, fn):
        super()._apply(fn)
        self.hash_offsets = fn(self.hash_offsets)
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        scalar_idx = (x + self.hash_offsets).view(batch_size, -1)
        scalars = self.scalars(scalar_idx).view(batch_size * seq_len, -1)
        indices = (x // self.ratio).view(batch_size * seq_len, -1)
        x = self.vectors(indices, per_sample_weights=scalars)
        x = x.view(batch_size, seq_len, -1)
        return x

    def init_weight_range(self, init_range):
        self.scalars.weight.data.fill_(1) #uniform_(-init_range, init_range)
        self.vectors.weight.data.uniform_(-init_range, init_range)

GLOVE_UNCASED_URL = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
GLOVE_FILENAME = "glove.42B.300d.txt"
BASE_FOLDER = "data/hashemb/"

class HashEmbeddingTrainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = HashEmbedding(self.cfg)
        self.embedding.init_weight_range(0.1)
        self.fc1 = nn.Linear(self.cfg['embedding-size'], 128, bias=False)
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

def load_glove(model):
    words, tensors = [], []
    lines = None
    download.download_and_extract(GLOVE_UNCASED_URL, "data/")
    with open(f'data/{GLOVE_FILENAME}', 'r', encoding='utf-8') as fip: # TODO make this download if it doesn't exist
        lines = fip.readlines()
        lines = lines[:1000000]
    for line in lines:
        line = line.strip().split()
        words.append(model.embedding.encode(line[0]))
        numbers = [float(x) for x in line[1:]]
        tensors.append(torch.FloatTensor(numbers))
    return torch.stack(words), torch.stack(tensors)

def train_on_glove(cfg, num_hashes=3, vocab_size=5000, embedding_dim=100, hash_ratio=10, use_gpu=True):
    device =torch.device('cuda') if use_gpu else torch.device('cpu')

    base_cfg = base.get_default_student_config('sst-2', None)
    base_cfg.update(cfg)
    cfg = base_cfg
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
            model.save_embeddings()
