from typing import List

from allennlp.data import Instance

from src.retrievers.retriever import Retriever


@Retriever.register("random")
class RandomRetriever(Retriever):

    def _retrieve_instances(self,
                            query_instance: Instance,
                            instances: List[str],
                            n_samples: int = None
                            ) -> List[str]:
        return self.random_sample(instances, n_samples)
    
    def get_last_scores_by_qids(self, qids: List[str]):
        scores = [1.0] * len(qids)
        return scores
