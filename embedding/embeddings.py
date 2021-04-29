from embedding.hash_emb import HashEmbedding
from embedding.bpe_emb import BPEmbedding
from embedding.char_emb import CharEmbedding
from embedding.cbow import CBOWEmbedding
from embedding.word2vec import Word2VecEmbedding

EMBEDDING_ZOO = {
    'bpe': BPEmbedding,
    'hash': HashEmbedding,
    'char': CharEmbedding,
    'cbow': CBOWEmbedding,
    "word2vec": Word2VecEmbedding
}

def get_embedding(cfg, load=True):
    # if cfg['embedding-type'] == 'cbow': # we have only trained cbow for these parameters
    #     if cfg['vocab-size'] != 5000:
    #         print(f"CBOW embedding fallback: vocab size from {cfg['vocab-size']} to 5000")
    #         cfg['vocab-size'] = 5000
    #     if cfg['embedding-dim'] != 16:
    #         print(f"CBOW embedding fallback: embedding dimension from {cfg['embedding-dim']} to 16")
    #         cfg['embedding-dim'] = 16

    try:
        return EMBEDDING_ZOO[cfg['embedding-type']](cfg, load)
    except FileNotFoundError: # Embeddings with specified vocab/dim not found.
        err = (
            f"Embeddings '{cfg['embedding-type']}' could not be found " +
            f"with vocab {cfg['vocab-size']} and embedding dim {cfg['embedding-dim']}!"
        )
        raise ValueError(err)
