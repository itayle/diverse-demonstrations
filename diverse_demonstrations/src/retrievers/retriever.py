from abc import abstractmethod
from random import Random
from typing import List, Dict

from allennlp.common import Registrable
from allennlp.data import Instance


class Retriever(Registrable):
    """
    indexes instances
    returns the most relevant instances for a given query instance
    """

    default_implementation = "random"
    _train_instances_per_qid = {}

    def __init__(self,
                 n_samples_in_prompt: int,
                 program_based: bool = True,
                 random_seed: int = 0,
                 prob_random: float = 0,
                 split:str = None,
                 instances_per_qid: Dict[str, List[Instance]] = None,
                 shuffle: bool = True,
                 ):
        self._random_seed = random_seed
        self._random = Random(random_seed)
        self._n_samples_in_prompt = n_samples_in_prompt
        self._program_based = program_based
        self._prob_random = prob_random
        self._split = split
        if self._split == "train":
            self._train_instances_per_qid.update(instances_per_qid)
            
        self._shuffle = shuffle

    @abstractmethod
    def retrieve_instances(self,
                           query_instance: Instance,
                           instances: List[str] = None,
                           n_samples: int = None
                           ) -> List[str]:
        if self._prob_random > 0:
            if self._random.random() < self._prob_random:
                return self.random_sample(instances, n_samples)
        return self._retrieve_instances(query_instance, instances, n_samples)
    
    def random_sample(self, instances, n_samples):
        n_samples = self._fix_n_samples(instances, n_samples)
        sampled_instances = self._random.sample(instances, k=n_samples)
        return sampled_instances

    @abstractmethod
    def _retrieve_instances(self,
                            query_instance: Instance,
                            instances: List[str],
                            n_samples: int = None) -> List[str]:
        raise NotImplementedError

    def _fix_n_samples(self,
                       instances: List,
                       n_samples: int = None
                       ) -> int:
        if not n_samples:
            n_samples = self._n_samples_in_prompt
        n_samples = min(n_samples, len(instances))
        return n_samples
    
    @abstractmethod
    def get_last_scores_by_qids(self, qids: List[str]):
        raise NotImplementedError    
