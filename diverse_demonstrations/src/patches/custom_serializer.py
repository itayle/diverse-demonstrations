from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import torch
import json
import numpy

from allennlp.common.util import sanitize
from allennlp.data.fields import TensorField
from allennlp.common import Registrable
from allennlp.data import DataLoader
from allennlp.evaluation.serializers.serializers import Serializer


def instance_separate(batch_size, outputs, skip_fields=None):
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


@Serializer.register("custom_serializer")
class CustomSerializer(Serializer):


    def _to_params(self) -> Dict[str, Any]:
        return {"type": "custom_serializer"}

    def __call__(
        self,
        batch: Dict[str, TensorField],
        output_dict: Dict,
        data_loader: DataLoader,
        output_postprocess_function: Optional[Callable] = None,
    ):
        """
        Serializer a batch.

        # Parameters

        batch: `Dict[str, TensorField]`
            The batch that was passed to the model's forward function.

        output_dict: `Dict`
            The output of the model's forward function on the batch

        data_loader: `DataLoader`
            The dataloader to be used.

        output_postprocess_function: `Callable`, optional (default=`None`)
            If you have a function to preprocess only the outputs (
            i.e. `model.make_human_readable`), use this parameter to have it
            called on the output dict.

        # Returns

        serialized: `str`
            The serialized batches as strings
        """
        if batch is None:
            raise ValueError("Serializer got a batch that is None")
        if output_dict is None:
            raise ValueError("Serializer got an output_dict that is None")

        # serialized = sanitize(batch)
        # if output_postprocess_function is not None:
        #     serialized.update(sanitize(output_postprocess_function(output_dict)))
        # else:
        #     serialized.update(sanitize(output_dict))
        serialized = ""
        batch_outputs = output_dict
        batch_size = max(len(f) if hasattr(f, "shape") and len(f.shape) > 0 else 0 for f in batch_outputs.values())
        batch_outputs = instance_separate(batch_size, batch_outputs,
                                               skip_fields=["predictions", "predicted_tokens", "log_probabilities"])
        for i, instance_output in enumerate(batch_outputs):
            if i > 0:
                serialized += "\n"
            serialized += json.dumps(sanitize(instance_output))

        return serialized


