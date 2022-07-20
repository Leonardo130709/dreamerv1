import argparse
import pathlib
import dataclasses
from .core import RLAlg
from .config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Additional arguments that match config fields name can be passed',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--load', type=pathlib.Path, help='path to the experiment dir')
    for field in dataclasses.fields(Config):
        parser.add_argument(f'--{field.name}',
                            type=int if field.type is tuple else field.type,
                            nargs="*" if field.type is tuple else '?',
                            default=field.default, help=str(field.type)
                            )
    return parser.parse_args()


def make_config(args):
    config = Config()
    fields = {k: v for k, v in vars(args).items() if k in vars(config).keys()}
    return dataclasses.replace(config, **fields)


def train():
    args = parse_args()
    if args.load:
        alg = RLAlg.load(args.load)
    else:
        config = make_config(args)
        alg = RLAlg(config)
    alg.learn()


if __name__ == "__main__":
    train()

