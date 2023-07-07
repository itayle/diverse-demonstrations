from abc import abstractmethod
from random import Random
from typing import List, Tuple, Dict, Union

import re

from allennlp.common import Registrable, Lazy
from allennlp.data import Instance, Token
from collections import defaultdict

from allennlp.data.fields import TextField

from src.retrievers.retriever import Retriever

class SelectionMethod(Registrable):
    default_implementation = "topk"

    _instances_per_split = defaultdict(list)
    _instance_per_qid = defaultdict(dict)

    def __init__(self,
                 is_test: bool,
                 instances: List[Instance],
                 retriever: Union[Retriever,Lazy[Retriever]],
                 split: str = None,
                 random_seed: int = 0,
                 n_samples_in_prompt: int = 1,
                 program_based: bool = True,
                 shuffle: bool = True,
                 sorting_retriever: Union[Retriever,Lazy[Retriever]] = None,
                 ):
        self._random_seed = random_seed
        self._random = Random(random_seed)
        self._is_test = is_test
        self._split = split
        self._shuffle = shuffle
        self._program_based = program_based

        assert n_samples_in_prompt >= 0
        self._n_samples_in_prompt = n_samples_in_prompt

        self._curr_split_instances = None

        if split is None:
            split = "test" if is_test else "train"
        self._split = split
        self._instances_per_split[self._split] = instances
        assert self._n_samples_in_prompt <= len(instances)

        self._instance_per_qid[self._split] = {ins['metadata']['ex_qid']: ins for ins in instances}

        if type(retriever) is Lazy:
            self._retriever = retriever.construct(
                n_samples_in_prompt=self._n_samples_in_prompt,
                program_based=self._program_based,
                split=self._split,
                instances_per_qid=self._instance_per_qid[self._split],
            )
        else:
            self._retriever = retriever

        if type(sorting_retriever) is Lazy:
            self._sorting_retriever = sorting_retriever.construct(
                n_samples_in_prompt=self._n_samples_in_prompt,
                program_based=self._program_based,
                split=self._split,
                instances_per_qid=self._instance_per_qid[self._split],
            )
        else:
            self._sorting_retriever = sorting_retriever

        if sorting_retriever and shuffle:
            print("A sorting retriever was given but shuffling is set to True - ignoring shuffle")
    
    def get_in_context_instances(self, instance: Instance) -> Tuple[List[Instance], Instance]:
        in_context_instances, test_instance = self._get_in_context_instances(instance)

        if self._sorting_retriever:
            in_context_instances_qid = [ex['metadata']['ex_qid'] for ex in in_context_instances]
            ordered_in_context_instances_qid = self._sorting_retriever.retrieve_instances(
                instance,
                in_context_instances_qid,
                n_samples=len(in_context_instances_qid)
            )
            in_context_instances = [self._instance_per_qid["train"][qid] for qid in ordered_in_context_instances_qid]
        elif self._shuffle:
            self._random.shuffle(in_context_instances)

        return in_context_instances, test_instance

    @abstractmethod
    def _get_in_context_instances(self, instance: Instance) -> Tuple[List[Instance], Instance]:
        raise NotImplementedError
