import argparse
import os
import sys

import numpy as np
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules
from dotenv import load_dotenv

sys.path.append('src')

load_dotenv()

WANDB_CALLBACKS = ["wandb", "wandb_best"]

def to_string(value):
    if isinstance(value, list):
        return [to_string(x) for x in value]
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)


def parse_args(argv):
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str)
    return args.parse_args(argv)


def get_wandb_callback(settings):
    callbacks = [cb for cb in settings['trainer']['callbacks'] if
                 type(cb) is dict and cb.get("type") in WANDB_CALLBACKS]
    return callbacks[0] if callbacks else None


def update_wandb_params(settings):
    wandb_callback = get_wandb_callback(settings)
    if not wandb_callback:
        return

    wandb_callback['wandb_kwargs']['resume'] = 'must'


def main_train(argv):
    args = parse_args(argv)
    import_module_and_submodules('src')

    experiment_dir = f"../runs/{args.name}/{args.name}"

    settings = Params.from_file(f"{experiment_dir}/config.json")
    print(f"Loading experiment from {experiment_dir}")
    print(f"path: {settings['train_data_path']}")
    
    os.environ["WANDB_RESUME"] = "must"

    train_model(
        params=settings,
        serialization_dir=experiment_dir,
        recover=True
    )


if __name__ == "__main__":
    main_train(sys.argv[1:])
