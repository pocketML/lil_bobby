from common import argparsers
from embedding import embeddings
from embedding import cbow
from embedding import word2vec

def train_embeddings(emb):
    pass

def main(args, sacred_experiment=None):
    cfg = {
        "task": args.task,
        "vocab-size": args.vocab_size,
        "context-size": args.context_size,
        "embedding-dim": args.embed_dim,
        "embedding-type": args.embed_type,
    }
    emb = embeddings.get_embedding(cfg, load=False)
    if args.embed_type == "cbow":
        cbow.train_embeddings(
            emb, args
        )
    elif args.embed_type == "word2vec":
        word2vec.train_embeddings(emb, args)
    else:
        raise ValueError("Embedding type not supported for pre-training (yet)")

if __name__ == "__main__":
    ARGS = argparsers.args_embeddings()
    main(ARGS)