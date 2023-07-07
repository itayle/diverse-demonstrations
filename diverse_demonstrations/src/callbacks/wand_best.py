import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING
from collections import deque

import torch

from allennlp.common import Params
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.log_writer import LogWriterCallback
from allennlp.training.callbacks.wandb import WandBCallback
from allennlp.training.util import get_train_and_validation_metrics, get_batch_size
from allennlp.data import TensorDict
from overrides import overrides

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

logger = logging.getLogger(__name__)


@TrainerCallback.register("wandb_best")
class WandBBestCallback(WandBCallback):
    """
        Logs training runs to Weights & Biases.
        Includes also best metrics

        !!! Note
            This requires the environment variable 'WANDB_API_KEY' to be set in order
            to authenticate with Weights & Biases. If not set, you may be prompted to
            log in or upload the experiment to an anonymous account.

    """

    def __init__(self, log_train_batches=True, log_last_train_batches=False, additional_config: Dict=None,
                 validation_metric: str = "accuracy", **kwargs) -> None:
        # fixes for starting multiple wandb runs from a single process
        if kwargs['wandb_kwargs'].get('reinit'):
            import wandb
            kwargs['wandb_kwargs']['settings'] = wandb.Settings(start_method="fork")

        super().__init__(**kwargs)
        recovered_wandb_folder = os.path.join(kwargs['serialization_dir'], "wandb")

        self._log_train_batches = log_train_batches
        self._log_last_train_batches = log_last_train_batches
        self._max_train_instances = 30
        self._train_instances_counter = 0
        if self._log_last_train_batches:
            self._train_outputs = deque([], maxlen=self._max_train_instances)
        else:
            self._train_outputs = []
        self._train_stats = []
        self._max_val_chosen_instances = 100
        self._chosen_instances_qid = []
        self._val_outputs = [[] for _ in range(self._max_val_chosen_instances)]

        self._stats_columns = ["n_actual_prompt_samples", "input_length"]
        self._train_columns = ["source_text", "gold_text", "meta_train_sources",
                               "meta_train_targets"]  # change self.table_name if you change columns
        self._val_columns = ["source_text", "gold vs predicted", "accuracy", "meta_train_sources",
                             "meta_train_targets"]  # change self.table_name if you change columns
        self._train_table_name = "train_v3"
        self._val_table_name = "preds_listv1"
        self._skip_choosing_instances = False

        self._additional_config = additional_config
        self._validation_metric = validation_metric

    @overrides
    def on_start(
            self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary=is_primary, **kwargs)

        if self._additional_config:
            self.wandb.config.update(self._additional_config)

    def on_batch(
            self,
            trainer: "GradientDescentTrainer",
            batch_inputs: List[TensorDict],
            batch_outputs: List[Dict[str, Any]],
            batch_metrics: Dict[str, Any],
            epoch: int,
            batch_number: int,
            is_training: bool,
            is_primary: bool = True,
            batch_grad_norm: Optional[float] = None,
            **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        num_batches = len(batch_inputs)
        batch_inputs = batch_inputs[0]
        batch_outputs = batch_outputs[0]
        batch_size = max(len(f) for f in batch_inputs.values())

        columns = self._train_columns if is_training else self._val_columns
        batch_outputs_sep, qids = self.instance_separate(batch_size, batch_outputs, columns)
        batch_stats_sep, _ = self.instance_separate(batch_size, batch_outputs, self._stats_columns)

        if not is_training or (epoch == 0 and self._log_train_batches):
            if is_training:
                for instance in batch_outputs_sep:
                    row_to_log = list(instance)
                    log_cond = self._log_last_train_batches or \
                               (not self._log_last_train_batches and self._train_instances_counter < self._max_train_instances)
                    if log_cond:
                        self._train_outputs.append(row_to_log)
                        self._train_instances_counter += 1
                    else:
                        self._log_train_batches = False
                        break
            else:
                assert (num_batches == 1)
                if not self._skip_choosing_instances:
                    for instance, qid in zip(batch_outputs_sep, qids):
                        self._chosen_instances_qid.append(qid)
                        if len(self._chosen_instances_qid) == self._max_val_chosen_instances:
                            self._skip_choosing_instances = True
                            break

                for instance, qid in zip(batch_outputs_sep, qids):
                    if qid in self._chosen_instances_qid:
                        row_index = self._chosen_instances_qid.index(qid)
                        row_to_log = list(instance)
                        self._val_outputs[row_index] = row_to_log

        if len(self._train_stats) < 150 and epoch == 0:
            for instance in batch_stats_sep:
                row_to_log = list(instance)
                self._train_stats.append(row_to_log)

    def on_epoch(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_primary: bool = True,
            **kwargs,
    ) -> None:
        """
        modified to include best metrics in val_metrics
        :param trainer:
        :param metrics:
        :param epoch:
        :param is_primary:
        :param kwargs:
        :return:
        """
        if not is_primary:
            return None
        assert self.trainer is not None

        train_metrics, val_metrics = get_train_and_validation_metrics(metrics)

        for key, value in metrics.items():
            if key.startswith("best_"):
                key = key.replace("validation_", "", 1)
                val_metrics[key] = value

        self.log_epoch(
            train_metrics,
            val_metrics,
            epoch,
        )
        # maybe redundant
        # https://docs.wandb.ai/guides/track/log#summary-metrics
        self.wandb.run.summary[f"best_{self._validation_metric}"] = metrics[f"best_validation_{self._validation_metric}"]

        if epoch == 0:
            model_results_filtered = list(filter(None, self._train_outputs))
            train_table = self.wandb.Table(columns=self._train_columns, data=model_results_filtered)
            self.wandb.log({self._train_table_name: train_table})

            stats_table = self.wandb.Table(columns=self._stats_columns, data=self._train_stats)
            self.wandb.log({
                'n_actual_prompt_samples': self.wandb.plot.histogram(stats_table, "n_actual_prompt_samples", title="Number of prompt instances"),
                'input_length': self.wandb.plot.histogram(stats_table, "input_length", title="Input length")},)

        if epoch == metrics["best_epoch"]:
            # filter empty rows. None values in columns are fine
            model_results_filtered = list(filter(None, self._val_outputs))
            val_table = self.wandb.Table(columns=self._val_columns, data=model_results_filtered)
            self.wandb.log({self._val_table_name: val_table})

        self._val_outputs = [None] * self._max_val_chosen_instances

    def instance_separate(self, batch_size, outputs, columns):
        instance_separated_output = [
            [None] * len(columns) for _ in range(batch_size)
        ]
        qids = []
        for name, output in list(outputs.items()):
            if name == "qid":
                qids = output
            if name not in columns:
                continue
            column_index = columns.index(name)
            if isinstance(output, torch.Tensor):
                # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                # This occurs with batch size 1, because we still want to include the loss in that case.
                if output.dim() == 0:
                    output = output.unsqueeze(0)

                if output.size(0) != batch_size:
                    continue
                output = output.detach().cpu().numpy()
            for instance_output, batch_element in zip(instance_separated_output, output):
                instance_output[column_index] = batch_element
        return instance_separated_output, qids
