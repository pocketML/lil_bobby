from common import argparsers
from compression.distillation import generate_loss

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    if args.generate_loss:
        generate_loss.generate(args)

if __name__ == "__main__":
    ARGS = argparsers.args_distill()

    main(ARGS)
