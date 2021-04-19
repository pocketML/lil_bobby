from embedding.hash_emb import HashEmbedding
from embedding.bpe_emb import BPEmbedding
from embedding.char_emb import CharEmbedding
from embedding.cbow import CBOWEmbedding

EMBEDDING_ZOO = {
    'bpe': BPEmbedding,
    'hash': HashEmbedding,
    'char': CharEmbedding,
    'cbow': CBOWEmbedding
}

def get_embedding(cfg):
    if cfg['embedding-type'] == 'cbow': # we have only trained cbow for these parameters
        if cfg['vocab-size'] != 5000 - 1:
            print(f"CBOW embedding fallback: vocab size from {cfg['vocab-size']} to 5000")
            cfg['vocab-size'] = 5000 - 1
        if cfg['embedding-dim'] != 16:
            print(f"CBOW embedding fallback: embedding dimension from {cfg['embedding-dim']} to 16")
            cfg['embedding-dim'] = 16

    return EMBEDDING_ZOO[cfg['embedding-type']](cfg)