from common import argparsers
from embedding import cbow

def main(args, sacred_experiment=None):
    # cbowemb = cbow.train_embeddings(
    #     args.context_size, args.task,
    #     args.embed_dim, args.vocab_size,
    #     args.epochs, args.batch_size, args.cpu
    # )
    cbow.load_pretrained_embeddings("sst-2", 16)

if __name__ == "__main__":
    ARGS = argparsers.args_cbow()
    main(ARGS)
