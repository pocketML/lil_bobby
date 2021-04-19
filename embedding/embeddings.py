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
    return EMBEDDING_ZOO[cfg['embedding-type']](cfg)