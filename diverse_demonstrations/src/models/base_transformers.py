from typing import Optional, Union, List, Dict, Tuple, Any, Type

from allennlp.common import cached_transformers
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.training.metrics.average import Average
from overrides import overrides


class Transformers(Model):
    def __init__(self, vocab: Vocabulary,
                 model_name: str,
                 beam_size: int = 3,
                 max_decoding_steps: int = 100,
                 extra_valid_accuracy_names: List[str] = None,
                 **kwargs) -> None:
        super().__init__(vocab, **kwargs)
        self._model_name = model_name

        # We only instantiate this when we need it.
        self._tokenizer: Optional[PretrainedTransformerTokenizer] = None

        self._max_decoding_steps = max_decoding_steps
        self._beam_size = beam_size

        self._metrics_names = ["accuracy"]
        self._metrics = {metric: Average() for metric in self._metrics_names}
        self._extra_valid_accuracy_names = extra_valid_accuracy_names or []
        self._metrics_extra_split = {extra_split: {metric: Average() for metric in self._metrics_names}
                                     for extra_split in self._extra_valid_accuracy_names}

    @property
    def tokenizer(self) -> PretrainedTransformerTokenizer:
        if self._tokenizer is None:
            self._tokenizer = cached_transformers.get_tokenizer(
                self._model_name, add_special_tokens=False
            )
        return self._tokenizer

    def update_metrics(self,
                       pred: str,
                       gold: str,
                       metadata: Dict[str, Any],
                       output_dict: Dict[str, Any],
                       ):
        results = {}
        results["accuracy"] = pred == gold

        for metric_name in results:
            metric_result = results[metric_name]
            if metadata['extra_eval_split']:
                self._metrics_extra_split[metadata['extra_eval_split']][metric_name](metric_result)
            else:
                self._metrics[metric_name](metric_result)
            if metric_name not in output_dict:
                output_dict[metric_name] = []
            output_dict[metric_name].append(metric_result)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {metric_name: self._metrics[metric_name].get_metric(reset=reset) for metric_name in self._metrics}
        if self._extra_valid_accuracy_names:
            metrics.update(
                {
                    f"{extra_valid_accuracy_name}_{metric_name}":
                    self._metrics_extra_split[extra_valid_accuracy_name][metric_name].get_metric(reset=reset)
                    for extra_valid_accuracy_name in self._extra_valid_accuracy_names
                    for metric_name in self._metrics_extra_split[extra_valid_accuracy_name]
                }
            )
        return metrics
