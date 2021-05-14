from embedding.hash_emb import HashEmbedding
from embedding.bpe_emb import BPEmbedding
from embedding.char_emb import CharEmbedding
from embedding.cbow import CBOWEmbedding
from embedding.word2vec import Word2VecEmbedding
from embedding.google_news import GoogleNewsEmb

EMBEDDING_ZOO = {
    'bpe': BPEmbedding,
    'hash': HashEmbedding,
    'char': CharEmbedding,
    'cbow': CBOWEmbedding,
    "word2vec": Word2VecEmbedding,
    'google': GoogleNewsEmb
}

def get_embedding(cfg, load=None):
    if load is None:
        if cfg['embedding-type'] in ['bpe', 'word2vec', 'cbow', 'hash']:
            load = True
        else:
            load = False
    try:
        return EMBEDDING_ZOO[cfg['embedding-type']](cfg, load)
    except FileNotFoundError: # Embeddings with specified vocab/dim not found.
        err = (
            f"Embeddings '{cfg['embedding-type']}' could not be found " +
            f"with vocab {cfg['vocab-size']} and embedding dim {cfg['embedding-dim']}!"
        )
        raise ValueError(err)
