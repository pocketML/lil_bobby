from abc import abstractmethod
import random
import os
import numpy as np
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
from common.task_utils import TASK_INFO
from compression.distillation import data
from compression.distillation import models

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

VOCAB_SIZE = 100_000

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
            try:
                embedding = np.array([float(value) for value in split[1:]])
            except ValueError:
                pass # This happens once or twice for some reason.
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

class Augmenter:
    @abstractmethod
    def augment(self, sentence):
        pass

class TinyBertAugmenter(Augmenter):
    def __init__(self):
        random.seed(1337)
        self.glove_normed, self.glove_vocab, self.glove_ids = load_glove()
        self.masked_lm = models.load_roberta_model('roberta_large')
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
        sent_candidates = []
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
                    new_sent[idx] = word_candidate.strip()

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

    base_path = f"{TASK_INFO[task]['path']}/distillation_data"

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    model = models.load_teacher(task)

    augmenter = classes[augment_technique]()

    training_data = data.load_train_data(task)
    sentence_pairs = len(training_data) == 3
    output_path = f"{base_path}/{augment_technique}.tsv"
    label_fn = lambda label: model.task.label_dictionary.string([label + model.task.label_dictionary.nspecial])

    prev_pct = 0
    print(f"Augmenting dataset: {prev_pct}% complete...", end="\r", flush=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        for index, train_example in enumerate(zip(*training_data)):
            sentences = [train_example[0]]
            if sentence_pairs: # Augment both sentences in sentence-pair classification.
                sentences.append(train_example[2])

            output_sentences = []

            for sent in sentences: # Augment each sentence (two sentences if sentence pairs task).
                output_sentences.append(augmenter.augment(sent.strip()))

            if sentence_pairs: # Create batch for prediction of label for new sentence.
                batch = collate_tokens(
                    [model.encode(sent1, sent2) for sent1, sent2 in zip(*output_sentences)],
                    pad_idx=1
                )
            else:
                batch = collate_tokens(
                    [model.encode(sent) for sent in output_sentences[0]],
                    pad_idx=1
                )

            batch_logits = model.predict( # Predict logits for new sentence.
                "sentence_classification_head", batch, return_logits=True
            )
            labels = [label_fn(label) for label in batch_logits.argmax(dim=1).tolist()]

            if sentence_pairs:
                iterator = zip(
                    output_sentences[0], output_sentences[1],
                    labels, batch_logits.tolist()
                )
                for sent1, sent2, target, logits in iterator:
                    logits_str = ','.join([str(x) for x in logits])
                    fp.write(f'{sent1.strip()}\t{sent2.strip()}\t{target.strip()}\t{logits_str}\n')
            else:
                iterator = zip(
                    output_sentences[0], labels, batch_logits.tolist()
                )
                for sent, target, logits in iterator:
                    logits_str = ','.join([str(x) for x in logits])
                    fp.write(f'{sent.strip()}\t{target.strip()}\t{logits_str}\n')

            pct_done = int((index / len(training_data[0])) * 100)
            if pct_done > prev_pct:
                print(f"Augmenting dataset: {pct_done}% complete...", end="\r", flush=True)
                prev_pct = pct_done

print()
