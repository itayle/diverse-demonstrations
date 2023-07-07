import argparse
import os
import sys
import time

import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)

from pprint import pprint
from random import Random
import numpy as np
import openai

from pathlib import Path
from typing import List

import pandas as pd
import wandb
from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.params import Params

from datasets import tqdm
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getcwd())
print(os.getcwd())

from src.dataset_readers.base_dataset_reader import BaseDatasetReader
from src.selection_methods.cover_ls import CoverLS
from src.selection_methods.topk import TopK
from src.selection_methods.dpp import DPP
from src.selection_methods.cover_source import CoverSource
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.random import RandomRetriever


SPLITS = {
    "smcalflow_cs": [(i, f"source_domain_with_target_num{i}", 5) for i in [8]], #[0, 8, 16, 32]
    # "smcalflow_cs_simple_v3": [(i, f"source_domain_with_target_num{i}") for i in [8,16]], #[8, 16, 32]
    "geo880_v2": [
        # ("length", "length", 1),
        ("stan", "standard", 1),
        # ("temp", "template_1", 1),
        # ("temp", "template_2", 1),
        # ("temp", "template_3", 1),
        # ("tmcd", "tmcd_1", 5),
        # ("tmcd", "tmcd_2", 1),
        # ("tmcd", "tmcd_3", 3),
    ],
    # "covr": [("covr3", f"cfg/seed_0/{split}", 5) for split in [43,100,25]]
    "covr": [("covr10", f"cfg/seed_0/{split}", 1) for split in [48, 25, 100, 110, 43, 115, 51, 8, 34, 99]]
}
# SPLITS["smcalflow_cs"].append(("iid","source_domain_with_target_num8", 1))
# SPLITS["smcalflow_cs"].append(("iid","source_domain_with_target_num16", 1))

print(f"cwd:{os.getcwd()}")
SMCALFLOW_TYPES = ["smcalflow_cs", "smcalflow_cs_simple_v3"]
DPP_TYPE = "DPP"
COVER_LS_TYPE = "cover_ls"
COVER_SOURCE_TYPE = "cover_source"
TOPK_BASELINE_TYPE = "topk"
RANDOM_BASELINE_TYPE = "random"

GPT_ENGINE = "text-davinci-002"
CODEX_ENGINE = "code-davinci-002"

all_experiments = []
conf_dict = {
    "NUM_TEST_INSTANCES_PER_SPLIT": 100
}
conf_dict["debug_mode"] = True
conf_dict["random_seed_options"]= [0]

conf_dict["ENGINE"] = CODEX_ENGINE
conf_dict["log_wandb"] = True
conf_dict["wandb_project"] = os.getenv("WANDB_PROJECT")
conf_dict["wandb_entity"] = os.getenv("WANDB_ENTITY")
conf_dict["wandb_group_name"] = "evaluate_NOFT"

conf_dict["dataset_options"] = ["geo880_v2",] # "geo880_v2" "smcalflow_cs" "covr"
conf_dict["n_samples_options"] = [24]
conf_dict["is_test_set"] = True
conf_dict["max_prompt_tokens"] = 7000
        
conf_dict["MAX_DECODING_STEPS"] = 500

conf_dict["order_by_bm25"] = True

conf_dict["ls_siblings"] = True

conf_dict["ls_size"] = 30
conf_dict["limit_ls_size_options"] = [30] #[1,2,30]
conf_dict["sampler_type"] = "deterministic"

conf_dict['phase_one_model'] = None
conf_dict['phase_one_model_split'] = None

for dataset in conf_dict["dataset_options"]:
    for selection_method in [COVER_LS_TYPE]:  # COVER_LS_TYPE, DPP_TYPE, COVER_SOURCE_TYPE
        for limit_ls_size in conf_dict["limit_ls_size_options"]:               
            splits = SPLITS[dataset]
            for split_group, split_name, best_beam_size in splits:
                iid_split = split_group == "iid"
                if dataset == "covr":
                    validation_split = "test"
                else:    
                    if conf_dict["is_test_set"]:
                        smcal_split = "test" if not iid_split else "test_s"
                        non_smcal_split = "test"
                    else:
                        smcal_split = "valid" if not iid_split else "valid_s"
                        non_smcal_split = "dev" if split_name != "standard" else "test"
                    validation_split = non_smcal_split if dataset not in SMCALFLOW_TYPES else smcal_split     
                
                # oracle - use the gold program labels instad of model predictions
                # predicted - use the predictions from the phase one model
                # source - use the utterance atoms instad of model predictions    
                # random - use a random retriever

                for atoms in ["predicted"]:  # "predicted", "oracle", "source" "random"
                    beam_size_options = [best_beam_size] if selection_method == COVER_LS_TYPE else [1]
                    for beam_size in beam_size_options:
                        for n_samples_in_prompt in conf_dict["n_samples_options"]:
                            curr_group = {}
                            curr_group_config = {
                                "dataset": dataset,
                                "split_name": split_name,
                                "split_group": split_group,
                                "atoms": atoms,
                                "selection_method": selection_method,
                                "limit_ls_size": limit_ls_size,
                                "validation_split": validation_split,
                                "iid_split": iid_split,
                                "beam_size": beam_size,
                                "n_samples_in_prompt": n_samples_in_prompt,
                            }
                            print(curr_group_config)
                            curr_group_str = "".join([f"{k}({v})" for k, v in curr_group_config.items()])
                            curr_group_experients = []
                            for random_seed in conf_dict["random_seed_options"]:
                                curr_exp = curr_group_config.copy()
                                curr_exp["random_seed"] = random_seed
                                curr_group_experients.append(curr_exp)
                            curr_group_entry = {
                                "group_str": curr_group_str,
                                "group_config": curr_group_config,
                                "experiments": curr_group_experients,
                            }
                            all_experiments.append(curr_group_entry)
print(f"Total of {len(all_experiments)} groups")
print(f"Total of {sum([len(group['experiments']) for group in all_experiments])} experiments")

tokenizer_params = {
    "type": "pretrained_transformer",
    "model_name": 'gpt2',
    "add_special_tokens": True
}
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_selection_method(exp_dict, train_instances):
    instances_per_qid = {ex['metadata']['ex_qid']: ex for ex in train_instances}
    key = exp_dict["selection_method"]
    n_samples_in_prompt = exp_dict["n_samples_in_prompt"]
    
    if exp_dict["atoms"] == "random":
        retriever = RandomRetriever(
        n_samples_in_prompt=n_samples_in_prompt,
        instances_per_qid=instances_per_qid,
        split="train",
        random_seed=exp_dict["random_seed"],
        )
    else:
        retriever = BM25Retriever(
            n_samples_in_prompt=n_samples_in_prompt,
            instances_per_qid=instances_per_qid,
            split="train",
            random_seed=exp_dict["random_seed"],
            program_based=(exp_dict["atoms"] != "source"),
            )
      
    if key == COVER_LS_TYPE:
        return CoverLS(
            is_test=True,
            split="train",
            n_samples_in_prompt=n_samples_in_prompt,
            instances=train_instances,
            retriever=retriever,
            limit_eval_ls_size=exp_dict["limit_ls_size"],
            random_seed=exp_dict["random_seed"],
        )
    elif key == COVER_SOURCE_TYPE:
        return CoverSource(
            is_test=True,
            split="train",
            n_samples_in_prompt=n_samples_in_prompt,
            instances=train_instances,
            retriever=retriever,
            random_seed=exp_dict["random_seed"],
        )
    
    elif key == TOPK_BASELINE_TYPE:
        return TopK(
            is_test=False,
            split="train",
            n_samples_in_prompt=n_samples_in_prompt,
            instances=train_instances,
            retriever=retriever,
            random_seed=exp_dict["random_seed"],
        )
    elif key == DPP_TYPE:
        return DPP(
            is_test=True,
            split="train",
            n_samples_in_prompt=n_samples_in_prompt,
            instances=train_instances,
            retriever=retriever,
            limit_eval_ls_size=exp_dict["limit_ls_size"],
            random_seed=exp_dict["random_seed"],
        )
    else:
        raise ValueError


def create_prompt(
        test_instance: Instance,
        prompt_instances: List[Instance],
        pre_source: str = "source:",
        pre_target: str = "target:",
):
    output = ""
    for ins in prompt_instances:
        source = ins['metadata']['source']
        target = ins['metadata']['target']
        output += f"{pre_source} {source}\n{pre_target} {target}\n"
    test_source = test_instance['metadata']['source']
    output += f"{pre_source} {test_source}\n{pre_target}"
    return output

def complete_prompt(prompt, engine_name, stop="source:", debug_mode=False):
     
    if debug_mode:
        output = {
            "text":"blah",
        }
    else:
        if engine_name == CODEX_ENGINE:
            requests_per_min = 5
            delay = 60.0 / requests_per_min
            time.sleep(delay)
        
        completion = openai.Completion.create(
            engine=engine_name,
            prompt=prompt,
            max_tokens=conf_dict["MAX_DECODING_STEPS"],
            stop=stop,
            temperature=0,
            logprobs=1
        )
        text = completion['choices'][0]['text'].strip()
        output = {
            "text": text,
        }

    return output


def openai_experiment(log_wandb: bool = False,):

    exp_list = [group_entry["experiments"] for group_entry in all_experiments]
    exp_list = [item for sublist in exp_list for item in sublist]
    accuracies = {}
   
    for exp_dict in exp_list:
        accuracy = run_experiments(exp_dict, log_wandb)
        accuracies[str(exp_dict)] = accuracy

    pprint(accuracies)
        
    
def run_experiments(exp_dict, log_wandb):
    dataset = exp_dict["dataset"]
    dataset_path = Path("../datasets", dataset, "all.jsonl")
    exp_split_name = exp_dict["split_name"]
    exp_split_path = Path("../datasets", dataset, "splits", f"{exp_split_name}/split.json")
    gpt_tokenizer = Tokenizer.from_params(params=Params(tokenizer_params))
      
    if exp_dict["atoms"] == "oracle": 
        phase_one_model = None
        phase_one_model_split = None
    else:
        phase_one_model = conf_dict['phase_one_model']
        phase_one_model_split = conf_dict['phase_one_model_split']
        assert phase_one_model is not None
    
    dataset_reader = BaseDatasetReader(
        split_path=exp_split_path,
        validation_split=exp_dict['validation_split'],
        ls_size=conf_dict["ls_size"],
        ls_siblings=conf_dict["ls_siblings"],
        phase_one_model=phase_one_model,
        phase_one_model_split=phase_one_model_split,
        phase_one_beam_size=exp_dict["beam_size"],
    )
    train_instances = list(dataset_reader.read(f"{dataset_path}"))
    test_instances = list(dataset_reader.read(f"@{dataset_path}"))
    Random(0).shuffle(test_instances)
    selection_method = get_selection_method(exp_dict, train_instances)
    predictions = []
    n_tested = 0
    n_correct = 0
    iid_str = "/iid" if exp_dict["iid_split"] else ""
    prompt_key = f"{dataset}/{exp_split_name}/{exp_dict['selection_method']}/{exp_dict['atoms']}/limit_ls_size({exp_dict['limit_ls_size']}){iid_str}"
    instances_per_qid = {ex['metadata']['ex_qid']: ex for ex in train_instances}
    # make order_retriever agnostic to ls_size of the selection_method
    if exp_dict["atoms"] == "random":
        order_retriever = RandomRetriever(
            n_samples_in_prompt=exp_dict["n_samples_in_prompt"],
            instances_per_qid=instances_per_qid,
            split="train",
            random_seed=exp_dict["random_seed"],
        )
    else:
        order_retriever = BM25Retriever(
            n_samples_in_prompt=exp_dict["n_samples_in_prompt"],
            instances_per_qid=instances_per_qid,
            split="train",
            random_seed=exp_dict["random_seed"],
            program_based=(exp_dict["atoms"] != "source"),
        )
            
    print(f"RUNNING EXPERIMENT {prompt_key} (seed:{exp_dict['random_seed']})")
    
    shortened = False
    for instance in tqdm(test_instances[:conf_dict["NUM_TEST_INSTANCES_PER_SPLIT"]], desc=prompt_key):
        in_context_instances = selection_method.get_in_context_instances(instance)[0]
        # reverse order so best is last
        if conf_dict["order_by_bm25"]:
            in_context_instances_qid = [ex['metadata']['ex_qid'] for ex in in_context_instances]
            ordered_in_context_instances_qid = order_retriever.retrieve_instances(
                instance,
                in_context_instances_qid,
                n_samples=len(in_context_instances_qid)
            )
            ordered_in_context_instances_qid = ordered_in_context_instances_qid[::-1]
            in_context_instances = [instances_per_qid[qid] for qid in ordered_in_context_instances_qid]
        prompt = create_prompt(instance, in_context_instances)        
        len_prompt_tokens = len(gpt_tokenizer.tokenize(prompt))
        while len_prompt_tokens >= conf_dict["max_prompt_tokens"]:
            # remove the first instance, which is the worst
            in_context_instances = in_context_instances[1:]
            prompt = create_prompt(instance, in_context_instances)
            len_prompt_tokens = len(gpt_tokenizer.tokenize(prompt))
            shortened = True
        model_output_dict = complete_prompt(prompt, conf_dict["ENGINE"], debug_mode=conf_dict["debug_mode"])
        model_output = model_output_dict["text"]
        gold_target = instance['metadata']['target']
        acc = model_output == gold_target
        
        curr_prediction = {
            'ex_qid': instance['metadata']['ex_qid'],
            'source': instance['metadata']['source'],
            'target': instance['metadata']['target'],
            'model_output': model_output,
            'prompt': prompt,
            'prompt_qids': [instance['metadata']['ex_qid'] for instance in in_context_instances],
            'acc': acc,
        }
        
        curr_prediction["len_prompt"] = len_prompt_tokens
        predictions.append(curr_prediction)

        n_tested += 1
        n_correct += int(acc)
    accuracy = n_correct / n_tested
    print(f"{prompt_key} accuracy: {accuracy:.2f}")

    results_df = pd.DataFrame(predictions)
    agg_results = results_df.mean(numeric_only=True).astype(float).to_frame().T
    agg_columns = agg_results.columns.tolist()
    round_dict = {col: 2 for col in agg_columns}
    agg_results = agg_results.round(round_dict).squeeze()
    agg_results["prompt_key"] = prompt_key
    agg_results["shortened"] = shortened
    print(agg_results)

    if log_wandb:
        wandb_config = dict(conf_dict)
        wandb_config.update(exp_dict)
        wandb.init(
            project=conf_dict["wandb_project"], 
            entity=conf_dict["wandb_entity"],
            name=f"{conf_dict['wandb_group_name']} {prompt_key.replace('/', '-')}",
            group=f"{conf_dict['wandb_group_name']}",
            reinit=True,
        )
        run_id = wandb.run.id
        wandb.config.update(wandb_config)
        wandb.log({"validation/best_accuracy": accuracy})
        wandb.log(agg_results.to_dict())
    
        wandb.finish() 
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_wandb', type=bool, default=conf_dict["log_wandb"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai_experiment(log_wandb=args.log_wandb)