from typing import List

import numpy as np

from allennlp.data import Instance
from rank_bm25 import BM25Okapi

from src.retrievers.retriever import Retriever


@Retriever.register("bm25_v2")
class BM25Retriever(Retriever):

    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if self._program_based:
            self._md_field = 'atoms_anonymized'
        else:
            self._md_field = 'source_atoms'

        instances = list(self._train_instances_per_qid.values())
        tokenized_corpus = [list(instance['metadata'][self._md_field]) for instance in instances]
        self._bm25 = BM25Okapi(tokenized_corpus)

        self._instances = [instance['metadata']['ex_qid'] for instance in instances]
        self._qid_to_doc_pos = {ins['metadata']['ex_qid']: i for i, ins in enumerate(instances)}
        self._last_retrieved_qids_to_scores = None

        self._cache = {}

    def _retrieve_instances(self,
                           query_instance: Instance,
                           instances: List[str],
                           n_samples: int = None,
                           ) -> List[str]:
        
        tokenized_query = tuple(query_instance['metadata'][self._md_field])

        n_samples = n_samples or self._n_samples_in_prompt

        if instances:
            if self._shuffle:
                self._random.shuffle(instances)
            if tokenized_query in self._cache:
                scores = self._cache[tokenized_query]
            else:
                scores = self._cache[tokenized_query] = self._bm25.get_scores(tokenized_query)
            doc_pos = [self._qid_to_doc_pos[qid] for qid in instances]
            relevant_scores = scores[doc_pos]

            top_n = np.argsort(relevant_scores)[::-1][:n_samples]
            retrieved_instances = [self._instances[doc_pos[pos]] for pos in top_n]
            self._last_retrieved_qids_to_scores = {retrieved_instances[i]: relevant_scores[score_index]
                                                   for i, score_index in enumerate(top_n)}
        else:
            retrieved_instances = self._bm25.get_top_n(tokenized_query, self._instances, n=n_samples)

        return retrieved_instances

    def get_last_scores_by_qids(self, qids: List[str]):
        #hacky way to get the scores
        scores = [self._last_retrieved_qids_to_scores[qid] for qid in qids]
        self._last_retrieved_qids_to_scores = None
        return scores
