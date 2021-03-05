from sacred import Experiment
from common.argparsers import args_experiment

def create_sacred_experiment(args):
    return Experiment(args.name)

if __name__ == "__main__":
    args = args_experiment()

    ex = create_sacred_experiment(args)
    ex.add_config(args.__dict__)
