import json
from typing import Dict, Any, List, Optional

import os

import numpy
import torch
from allennlp.common.util import sanitize

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data import TensorDict


@TrainerCallback.register("save_predictions")
class SavePredictionsCallback(TrainerCallback):
    def __init__(self, serialization_dir: str):
        super().__init__(serialization_dir)

        self._all_outputs = []
        self._predictor = None
        self._serialization_dir = serialization_dir

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if epoch == metrics["best_epoch"]:
            fw = open(os.path.join(self._serialization_dir, "best_outputs.jsonl"), "wt")
            for output in self._all_outputs:
                fw.write(json.dumps(sanitize(output)) + "\n")

        self._all_outputs = []

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        if is_training:
            return

        assert(len(batch_inputs) == 1)
        batch_inputs = batch_inputs[0]
        batch_outputs = batch_outputs[0]
        batch_size = max(len(f) for f in batch_inputs.values())

        batch_outputs = self.instance_separate(batch_size, batch_outputs, skip_fields=["predictions", "predicted_tokens", "log_probabilities"])
        self._all_outputs += batch_outputs

    def instance_separate(self, batch_size, outputs, skip_fields=None):
        instance_separated_output: List[Dict[str, numpy.ndarray]] = [
            {} for _ in range(batch_size)
        ]
        for name, output in list(outputs.items()):
            if skip_fields and name in skip_fields:
                continue
            if isinstance(output, torch.Tensor):
                # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                # This occurs with batch size 1, because we still want to include the loss in that case.
                if output.dim() == 0:
                    output = output.unsqueeze(0)

                if output.size(0) != batch_size:
                    continue
                output = output.detach().cpu().numpy()
            for instance_output, batch_element in zip(instance_separated_output, output):
                instance_output[name] = batch_element
        return instance_separated_output
