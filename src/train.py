from .core import Config, Dreamer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='walker_stand')
    parser.add_argument('--observe', default='pixels')
    parser.add_argument('--logdir', default='logdir')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def make_config(args):
    config = Config()
    config.task = args.task
    config.observe = args.observe
    config.device = args.device
    config.logdir = args.logdir
    return config

def train():
    args = parse_args()
    config = make_config(args)
    dreamer = Dreamer(config)
    dreamer.learn()
    return 0

if __name__ == "__main__":
    train()

