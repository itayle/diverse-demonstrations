import json
import os
from typing import List
from allennlp.data import Instance

from src.retrievers.retriever import Retriever

@Retriever.register("sbert")
class SBertRetriever(Retriever):

    def __init__(self,
                 retrieval_file: str = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        assert self._program_based == False
        instances = list(self._train_instances_per_qid.values())        
        self._instances = [instance['metadata']['ex_qid'] for instance in instances]
        
        self._retrieved_instances_per_qid = {}
        if os.path.exists(retrieval_file):
            print(f"Loading retrieval scores from {retrieval_file}")
            with open(retrieval_file, 'r') as fp:
                self._retrieved_instances_per_qid.update(json.load(fp))
        else:
            raise FileNotFoundError(f"Retrieval scores file {retrieval_file} does not exist")
        
        self._last_retrieved_qids_to_scores = None

        self._cache = {}

    def _retrieve_instances(self,
                           query_instance: Instance,
                           instances: List[str],
                           n_samples: int = None,
                           ) -> List[str]:
        
        query_qid = query_instance['metadata']['ex_qid']

        n_samples = n_samples or self._n_samples_in_prompt

        if instances:
            if self._shuffle:
                self._random.shuffle(instances)
            if query_qid in self._cache:
                scores = self._cache[query_qid]
            else:
                scores = self._cache[query_qid] = self._retrieved_instances_per_qid[query_qid]
            relevant_scores = [score_entry for score_entry in scores if score_entry['qid'] in instances]
            top_n = relevant_scores[:n_samples] # we assume the scores are already sorted
            retrieved_instances = [score_entry['qid'] for score_entry in top_n]
            self._last_retrieved_qids_to_scores = {score_entry['qid']: float(score_entry['score']) for score_entry in top_n}
        else:
            raise NotImplementedError

        return retrieved_instances
    
    def get_last_scores_by_qids(self, qids: List[str]):
        #hacky way to get the scores
        scores = [self._last_retrieved_qids_to_scores[qid] for qid in qids]
        self._last_retrieved_qids_to_scores = None
        return scores
