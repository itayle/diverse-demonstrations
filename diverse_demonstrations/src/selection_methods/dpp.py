import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple

from allennlp.data import Instance

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from submodlib import LogDeterminantFunction

from src.selection_methods.selection_method import SelectionMethod


@SelectionMethod.register("dpp")
class DPP(SelectionMethod):

    def __init__(self,
                 is_test: bool,
                 instances: List[Instance],
                 limit_eval_ls_size: float = None,
                 dpp_mode: str = "write",
                 read_file_path: str = None,
                 **kwargs):
        super().__init__(is_test, instances, **kwargs)
        assert is_test
        self._dpp_mode = dpp_mode
        if self._dpp_mode == "read":
            assert read_file_path is not None
            df = pd.read_csv(read_file_path)
            df = df[["ex_qid", "prompt_qids"]]
            self._prompt_qids_per_qid = {qid: eval(prompt_qids) for qid, prompt_qids in df.values}
        else:            
            self._limit_eval_ls_size = limit_eval_ls_size
            self._ls_field = "ls_by_size_flat"
            self._ls_field_by_size = "ls_by_size"
            
            ls_pool_train_list = []
            self._train_qids = []
            self._ls_to_int = {}
            if self._shuffle:
                    self._random.shuffle(instances) #shuffle for random seed effect on phi order
            for i, instance in enumerate(instances):
                ls_pool_by_size = [(ls_size, ls_pool) for ls_size, ls_pool in enumerate(instance["metadata"][self._ls_field_by_size])
                            if ls_pool]
                if self._limit_eval_ls_size:
                    ls_pool_by_size = [(ls_size, ls_pool) for ls_size, ls_pool in ls_pool_by_size if ls_size < self._limit_eval_ls_size]
                if not ls_pool_by_size:
                    raise NotImplementedError
                
                flat_ls_pool = [ls for ls_size, ls_pool in ls_pool_by_size for ls in ls_pool]
                for ls in flat_ls_pool:
                    if ls not in self._ls_to_int:
                        self._ls_to_int[ls] = len(self._ls_to_int)
                
                # countvectorizer has weird behavior with raw ls strings, so we use ints. also weird with short strings
                instance_ls_str = " ".join([f"ls{self._ls_to_int[ls]}" for ls in flat_ls_pool])               
                
                ls_pool_train_list.append(instance_ls_str)
                self._train_qids.append(instance["metadata"]["ex_qid"]) #also shuffled
                
            pipe = Pipeline([('count', CountVectorizer()),('tfid', TfidfTransformer())]).fit(ls_pool_train_list)
            
            phi = pipe.transform(ls_pool_train_list).toarray()
            self._S = phi @ phi.T
            # all_features = pipe['count'].get_feature_names_out()
            

    def _get_in_context_instances(self, instance: Instance) -> Tuple[List[Instance], Instance]:
        """
        :return: list of in-context samples for each instance in batch
        """
        in_context_instances = []
        if self._dpp_mode == "read":
            greedy_qids = self._prompt_qids_per_qid[instance["metadata"]["ex_qid"]]
        else:
            n_train_samples = len(self._train_qids)
            ordered_train_qids = self._retriever.retrieve_instances(instance,
                                                                    list(self._train_qids),
                                                                    n_samples=n_train_samples)
            quality_scores = self._retriever.get_last_scores_by_qids(qids=list(self._train_qids)) #ordered according to phi
            q = np.array(quality_scores)
            q /= q.max()

            K = q * self._S * q
            obj_log_det = LogDeterminantFunction(n=n_train_samples,
                                                mode="dense",
                                                lambdaVal=0,
                                                sijs=K)
            greedy_indices_and_scores = obj_log_det.maximize(budget=self._n_samples_in_prompt,
                                                                optimizer='NaiveGreedy',
                                                                stopIfZeroGain=False,
                                                                stopIfNegativeGain=False,
                                                                verbose=False)
            greedy_indices, greedy_scores = zip(*greedy_indices_and_scores)
            greedy_qids = [self._train_qids[i] for i in greedy_indices]
        in_context_instances = [self._instance_per_qid["train"][qid] for qid in greedy_qids]

        return in_context_instances, instance
