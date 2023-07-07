import os
import sys
import argparse
import time
import numpy as np
import copy
from allennlp.common.params import with_overrides, parse_overrides
from dotenv import load_dotenv
from allennlp.commands.train import train_model
from allennlp.common import Params

from allennlp.common.util import import_module_and_submodules

load_dotenv()
sys.path.append('src')
sys.path.append(os.getcwd())

WANDB_CALLBACKS = ["wandb", "wandb_best"]
BASE_CONFIG_FILE = "../configs/base.jsonnet"

def to_string(value):
    if isinstance(value, list):
        return [to_string(x) for x in value]
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)


def parse_args(argv):
    random_seed = np.random.randint(9999)
    args = argparse.ArgumentParser()
    args.add_argument('name', nargs='?', default='')
    args.add_argument('--model-name', default='t5-large')
    args.add_argument('--selection_method', choices=['topk', 'cover_ls'], default='topk')
    args.add_argument('--dataset', choices=['covr', 'smcalflow_cs', 'smcalflow_cs_simple_v3', 'geo880_v2'], required=True)
    args.add_argument("--split-name", default="", type=str)
    args.add_argument('--recover', action='store_true')
    args.add_argument('--recover_name', default="", type=str)
    args.add_argument("--gpu", default="0", type=str)
    args.add_argument("--batch_size", default=16, type=int)
    args.add_argument("--val_batch_size", default=None, type=int)
    args.add_argument("--epochs", default=15, type=int)
    args.add_argument("--n_samples_in_prompt", default=6, type=int)
    args.add_argument("--wandb_group", default="", type=str)
    args.add_argument("--wandb_project", default="", type=str)
    args.add_argument("-o", "--overrides", default="", type=str)
    args.add_argument("--seed", default=random_seed, type=int)
    args.add_argument("--retriever", choices=['random', 'bm25_v2'], default='random', type=str)
    args.add_argument("--eval-retriever", choices=['random', 'bm25_v2'], default=None, type=str)
    args.add_argument("--sort-retriever", choices=['random', 'bm25_v2'], default=None, type=str)
    args.add_argument("--validation_split", default=None)
    args.add_argument("--extra_eval_split", default=None)
    args.add_argument("--validate_every", type=int, default=None)
    args.add_argument("--lr", type=float, default=1e-5)

    return args.parse_args(argv)


def get_experiment_name(args, debug_mode=False):
    if args.recover and args.recover_name:
        return args.recover_name
    if debug_mode:
        return 'debug'
    
    name = "_".join(
        [
            f"{key}({value})"
            for key, value in vars(args).items()
            if value is not None
               and key in ["split_name", "dataset"]
        ]
    )
    name += f"_trim({np.random.randint(9999)})"
    return name


def disable_wandb(settings):
    settings['trainer']['callbacks'] = [cb for cb in settings['trainer']['callbacks'] if
                                        type(cb) is dict and cb.get("type") not in WANDB_CALLBACKS]


def get_wandb_callback(settings):
    callbacks = [cb for cb in settings['trainer']['callbacks'] if type(cb) is dict and cb.get("type") in WANDB_CALLBACKS]
    return callbacks[0] if callbacks else None


def update_wandb_params(settings, experiment_name, args, additional_wandb_config):
    wandb_callback = get_wandb_callback(settings)
    if not wandb_callback:
        return
    wandb_callback['name'] = experiment_name
    # add more information to wandb config which is not in the original experiment config
    wandb_callback["additional_config"] = additional_wandb_config

    wandb_callback['wandb_kwargs'] = dict()
    wandb_callback['wandb_kwargs']['job_type'] = f"{args.n_samples_in_prompt}_samples_in_prompt"


def fix_trainer_callbacks(args, serialization_dir, settings):
    if args.recover:
        recovered_config_file = os.path.join(serialization_dir, "config.json")
        loaded_params = Params.from_file(recovered_config_file)
        loaded_callbacks = copy.deepcopy(loaded_params.params['trainer']['callbacks'])
        settings['trainer']['callbacks'] = loaded_callbacks


def merge_settings(base, to_merge, override):
    output = with_overrides(base, to_merge)
    output = with_overrides(output, parse_overrides(override))
    return output


def set_retriever_settings(settings, sort_retriever):
    if sort_retriever:
        settings['data_loader']['selection_method']['sorting_retriever'] = {'type': sort_retriever, 'program_based': True}

def set_training_settings(settings, args):
    if args.validate_every is not None:
        settings['trainer']['validate_every'] = args.validate_every

    # we don't have dev set for covr, so we should save last epoch
    if "covr" in args.dataset:
        settings['trainer']['checkpointer']['save_last_epoch'] = True


def set_dataset_settings(dataset,
                         split_name=None,
                         settings=None,
                         additional_wandb_config=None,
                         validation_split=None,
                         ):
    wandb_config = {"dataset": dataset}
    dataset_settings = {}
    
    if split_name:
        dataset_settings["train_data_path"] = f"../datasets/{dataset}/all.jsonl"
        if split_name:
            wandb_config["split_name"] = split_name
            dataset_settings["validation_data_path"] = f"@../datasets/{dataset}/all.jsonl"
            dataset_settings[
                "dataset_reader.split_path"] = f"../datasets/{dataset}/splits/{split_name}/split.json"
    else:
        dataset_settings["train_data_path"] = f"../datasets/{dataset}/train.jsonl"
        dataset_settings["validation_data_path"] = f"@../datasets/{dataset}/test.jsonl"

    if settings:
        settings.update(with_overrides(settings, dataset_settings))
    if additional_wandb_config:
        additional_wandb_config.update(wandb_config)

    if validation_split is None:
        if "smcalflow_cs" not in dataset:
            # default validation split is called "test"
            settings["dataset_reader"]["validation_split"] = "test"
        else:
            raise ValueError("For smcalflow-cs please explicitly define the validation split valid/test")
    else:
        settings["dataset_reader"]["validation_split"] = validation_split

    extra_valid_accuracy_names = []

    if "smcalflow_cs" in dataset:
        extra_eval_split = f"{validation_split}_s"
        settings["dataset_reader"]["extra_eval_split"] = extra_eval_split
        extra_valid_accuracy_names.append(extra_eval_split)

    settings["model"]["extra_valid_accuracy_names"] = extra_valid_accuracy_names

    return dataset_settings, wandb_config


def main_train(argv):
    args = parse_args(argv)
    import_module_and_submodules('src')

    jsonnet_ext_vars = {}
    for key, value in vars(args).items():
        jsonnet_ext_vars[key] = to_string(value)

    settings = Params.from_file(BASE_CONFIG_FILE, ext_vars=jsonnet_ext_vars).params

    additional_wandb_config = {"args": vars(args)}

    set_dataset_settings(
        dataset=args.dataset,
        split_name=args.split_name,
        settings=settings,
        additional_wandb_config=additional_wandb_config,
        validation_split=args.validation_split,
    )
    set_retriever_settings(settings, args.sort_retriever)
    set_training_settings(settings, args)
    settings = with_overrides(settings, parse_overrides(args.overrides))

    # validation_data_loader should be copied after all settings were set otherwise there will be a diff between train/val load
    settings['validation_data_loader'] = copy.deepcopy(settings['data_loader'])
    if args.val_batch_size and args.batch_size != args.val_batch_size:
        settings['validation_data_loader']['batch_size'] = args.val_batch_size
    if args.eval_retriever and args.eval_retriever != args.retriever:
        settings['validation_data_loader']['selection_method']['retriever']['type'] = args.eval_retriever

    force = False
    debug_mode = (sys.gettrace() is not None) or True
    if debug_mode:
        force = True
        disable_wandb(settings)

    experiment_name = get_experiment_name(args, debug_mode)
    print(f"experiment name:{experiment_name}")

    serialization_dir_path = get_serialization_dir_path(experiment_name, debug_mode, recover=args.recover)

    update_wandb_params(settings, experiment_name, args, additional_wandb_config)
    fix_trainer_callbacks(args, serialization_dir=serialization_dir_path, settings=settings)

    train_model(
        params=Params(settings),
        serialization_dir=serialization_dir_path,
        recover=args.recover,
        force=force
    )


def get_serialization_dir_path(experiment_name, debug_mode, parent_experiments_path="../runs", recover=False):
    serialization_dir = experiment_name.replace('/', '-')
    serialization_dir_path = os.path.join(parent_experiments_path, serialization_dir).replace(" ", "_")
    if not debug_mode and not recover:
        if os.path.exists(serialization_dir_path):
            serialization_dir += '_' + str(time.time()).replace('.', '')
            serialization_dir_path = os.path.join(parent_experiments_path, serialization_dir).replace(" ", "_")
    return serialization_dir_path


if __name__ == "__main__":
    main_train(sys.argv[1:])
