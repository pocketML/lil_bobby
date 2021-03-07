from common import argparsers

def main(args, sacred_experiment=None):
    print(f"Doing some sick compression with args: {args}")

if __name__ == "__main__":
    ARGS = argparsers.args_compress()

    main(ARGS)
