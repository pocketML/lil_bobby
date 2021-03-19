from abc import abstractmethod
import random
import os
import numpy as np
from fairseq.models.roberta import RobertaModel
from common.task_utils import TASK_INFO
from compression.distillation.data import load_train_data

STOP_WORDS = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "'s", "'re"
]

VOCAB_SIZE = 10_000

def load_glove():
    words = []
    embeddings = {}

    with open("data/glove.840B.300d.txt", "r", encoding="utf-8") as fp:
        for count, line in enumerate(fp):
            if count == VOCAB_SIZE:
                break
            split = line.split()
            word = split[0]
            words.append(word)
            embedding = np.array([float(value) for value in split[1:]])
            embeddings[word] = embedding

    vocab = {w: idx for idx, w in enumerate(words)}
    ids_to_tokens = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(embeddings[ids_to_tokens[0]])
    emb_matrix = np.zeros((VOCAB_SIZE, vector_dim))
    for word, v in embeddings.items():
        if word == '<unk>':
            continue
        emb_matrix[vocab[word], :] = v

    # normalize each word vector
    d = (np.sum(emb_matrix ** 2, 1) ** 0.5)
    emb_norm = (emb_matrix.T / d).T
    return emb_norm, vocab, ids_to_tokens

def load_language_model(task):
    # Might be used later
    data_path = "."
    if task is not None:
        data_path = f'{TASK_INFO[task]["path"]}/processed/{task}-bin/'
    model = None
    model = RobertaModel.from_pretrained(
        'models/roberta.base',
        checkpoint_file="model.pt",
        data_name_or_path=data_path
    )
    model.eval()
    return model

class Augmenter:
    @abstractmethod
    def augment(self, sentence):
        pass

class TinyBertAugmenter(Augmenter):
    def __init__(self):
        self.masked_lm = load_language_model(None)
        self.glove_normed, self.glove_vocab, self.glove_ids = load_glove()
        # Initialize augment parameters.
        self.p_threshold = 0.4 # Threshold probability.
        self.n_samples = 20 # Number of augmented samples per examples.
        self.k_candidates = 15 # Size of candidate set.

    def masked_lm_predict(self, tokens):
        predictions = self.masked_lm.fill_mask(" ".join(tokens), topk=self.k_candidates)
        return [predict_tuple[2] for predict_tuple in predictions]

    def word_distance(self, word):
        if word not in self.glove_vocab:
            return []

        word_idx = self.glove_vocab[word]
        word_emb = self.glove_normed[word_idx]

        dist = np.dot(self.glove_normed, word_emb.T)
        dist[word_idx] = -np.Inf

        candidate_ids = np.argsort(-dist)[:self.k_candidates]
        return [self.glove_ids[idx] for idx in candidate_ids][:self.k_candidates]

    def augment_word(self, sentence, mask_token_idx, mask_token):
        sentence_split = sentence.split(" ")
        tokenized_len = len(sentence_split)

        word_piece_ids = []

        for token_idx in range(tokenized_len):
            word_piece = self.masked_lm.encode(sentence_split[token_idx])
            if len(word_piece) == 3: # Word is not split by BPEmb.
                if token_idx < mask_token_idx:
                    word_piece_ids = []
                elif token_idx == mask_token_idx:
                    word_piece_ids = [token_idx]
                else:
                    break
            else:
                word_piece_ids.append(token_idx)

        word_candidates = []

        if len(word_piece_ids) == 1: # Predict with masked LM.
            sentence_split[word_piece_ids[0]] = '<mask>'
            word_candidates = self.masked_lm_predict(sentence_split)
        else: # Use GLoVe distances.
            word_candidates = self.word_distance(mask_token)

        if word_candidates == []:
            word_candidates.append(mask_token)

        return word_candidates

    def augment(self, sentence):
        tokens = sentence.split(" ")
        sent_candidates = [sentence]
        word_candidates = {}
        for (idx, word) in enumerate(tokens):
            if word not in STOP_WORDS:
                word_candidates[idx] = self.augment_word(sentence, idx, word)

        for _ in range(self.n_samples):
            new_sent = list(tokens)

            for idx in word_candidates:
                word_candidate = random.choice(word_candidates[idx])

                x = random.random()
                if x < self.p_threshold:
                    new_sent[idx] = word_candidate

            if " ".join(new_sent) not in sent_candidates:
                sent_candidates.append(" ".join(new_sent))

        return sent_candidates

class MaskAugmenter(Augmenter):
    def augment(self, sentence):
        pass

class PoSAugmenter(Augmenter):
    def __init__(self):
        self.pos_tagger = None # Initialize PoS tagger.

    def augment(self, sentence):
        pass

class NGramAugmenter(Augmenter):
    def augment(self, sentence):
        pass

def augment(task, augment_technique):
    classes = {
        "tinybert": TinyBertAugmenter, "mask": MaskAugmenter,
        "pos": PoSAugmenter, "ngram": NGramAugmenter
    }

    base_path = f"{TASK_INFO[task]['path']}/augmented_data"

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    augment_class = classes[augment_technique]()

    training_data = load_train_data(task)
    sentence_pairs = len(training_data) == 3
    output_path = f"{base_path}/{augment_technique}.tsv"
    max_stuff = 0
    with open(output_path, "w", encoding="utf-8") as fp:
        for train_example in zip(*training_data):
            sentences = [train_example[0]]
            label = train_example[1]
            if sentence_pairs: # Augment both sentences in sentence-pair classification.
                sentences.append(train_example[2])

            augmented_sentence_output = []
            
            for sent in sentences:
                augmented_sentences = augment_class.augment(sent.strip())
                augmented_sentence_output.append(augmented_sentences)

            for sample in augmented_sentence_output[0]:
                fp.write(f"{sample}\n")
            max_stuff += 1
            if max_stuff == 10:
                break
