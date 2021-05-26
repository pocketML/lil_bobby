import pickle
import unicodedata

import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec as Word2Vec

from common import data_utils
from common.model_utils import get_model_path
from embedding.base import Embedding

class Word2VecEmbedding(Embedding):
    def __init__(self, cfg, load=True):
        super().__init__(cfg, load)

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

    def create_vocab(self, loaded_data):
        tokenized_data = [self.tokenize(sent) for sent in loaded_data]
        self.model.build_vocab(tokenized_data)
        vocab = list(word for word in self.model.wv.vocab)
        self.mapping = {word: i + len(self.specials) for i, word in enumerate(vocab)}
        return tokenized_data

    def save(self, task):
        model_path = get_model_path(task, "embeddings")

        vocab = self.cfg['vocab-size']
        dim = self.cfg['embedding-dim']

        self.model.save(f"{model_path}/word2vec_{vocab}_{dim}_model.model")

        with open(f"{model_path}/word2vec_{vocab}_{dim}_dict.bin", "wb") as fp:
            pickle.dump(self.mapping, fp)

    def init_embeddings(self):
        self.specials = {'unknown': 0, 'pad': 1}

        vocab = self.cfg['vocab-size']
        dim = self.cfg['embedding-dim']

        if self.load_pretrained:
            model_path = get_model_path(self.cfg["task"], "embeddings")
            self.model = Word2Vec.load(f"{model_path}/word2vec_{vocab}_{dim}_model.model")
            vectors = self.model.wv

            with open(f"{model_path}/word2vec_{vocab}_{dim}_dict.bin", "rb") as fp:
                self.mapping = pickle.load(fp)

            embeddings = torch.tensor([vectors.get_vector(word) for word in self.mapping])
            if embeddings.shape[0] < self.cfg["vocab-size"]+1:
                pad_amount = (self.cfg["vocab-size"] + 1) - embeddings.shape[0]
                embeddings = torch.nn.functional.pad(embeddings, pad=(0, 0, pad_amount, 0), mode="constant", value=self.specials['pad'])

            embedding = nn.Embedding.from_pretrained(embeddings)
        else:
            self.model = Word2Vec(
                size=self.cfg["embedding-dim"],
                window=self.cfg["context-size"],
                min_count=1,
                max_final_vocab=self.cfg["vocab-size"]
            )
            embedding = nn.Embedding(self.cfg["vocab-size"]+1, self.cfg['embedding-dim'])

        return embedding

    def encode(self, sentence):
        return torch.LongTensor([self.mapping.get(w, self.specials['unknown']) for w in self.tokenize(sentence)])

def train_embeddings(word2vec, args):
    train_data = data_utils.load_train_data(args.task)
    loaded_data = train_data[0] if len(train_data) == 2 else train_data[0] + train_data[2]
    if not args.only_original_data:
        augment_data = data_utils.load_augment_data(args.task, "tinybert")
        augment_data = augment_data[0] if len(augment_data) == 1 else augment_data[0] + augment_data[1]
        loaded_data = loaded_data + augment_data
    print("Data loaded...")

    tokenized_data = word2vec.create_vocab(loaded_data)
    num_tokens = sum(len(sent) for sent in tokenized_data)

    print(f"Sentences: {len(tokenized_data)}")
    print(f"Tokens: {num_tokens}")

    word2vec.model.train(
        tokenized_data,
        total_words=sum(len(sent) for sent in tokenized_data),
        epochs=args.epochs,
        compute_loss=True
    )
    final_loss = word2vec.model.get_latest_training_loss()

    print(word2vec.model.most_similar("cat"))

    print(word2vec.model.wv.get_vector("cat"))

    print(f"Loss: {final_loss}")
    word2vec.save(args.task)
