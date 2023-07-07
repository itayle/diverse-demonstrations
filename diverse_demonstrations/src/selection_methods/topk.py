from typing import List, Tuple

from allennlp.data import Instance

from src.selection_methods.selection_method import SelectionMethod


@SelectionMethod.register("topk")
class TopK(SelectionMethod):
    def _get_in_context_instances(self, instance: Instance) -> Tuple[List[Instance], Instance]:
        if self._n_samples_in_prompt == 0:
            return [], instance
        
        # sort for random reproducibility
        sampling_candidates_qid = list(self._instance_per_qid['train'].keys())
        if instance['metadata']['ex_qid'] in sampling_candidates_qid:
            sampling_candidates_qid.remove(instance['metadata']['ex_qid']) 
        sampled_instances_qids = self._retriever.retrieve_instances(instance, sampling_candidates_qid)
        sampled_instances = [self._instance_per_qid["train"][qid] for qid in sampled_instances_qids]

        return sampled_instances, instance
