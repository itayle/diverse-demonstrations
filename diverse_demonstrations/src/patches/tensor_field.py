from typing import Dict, Union, Optional, Any

import numpy as np
import torch
from allennlp.data.fields import TensorField as OriginalTensorField


class TensorField(OriginalTensorField):
    """
    Same as TensorField, except that we can control left/right padding
    """

    def __init__(
            self,
            *args,
            padding_on_right: bool = True,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._padding_on_right = padding_on_right

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        tensor = self.tensor
        while len(tensor.size()) < len(padding_lengths):
            tensor = tensor.unsqueeze(-1)
        if self._padding_on_right:
            pad = [
                padding
                for i, dimension_size in reversed(list(enumerate(tensor.size())))
                for padding in [0, padding_lengths["dimension_" + str(i)] - dimension_size]
            ]
        else:
            pad = [
                padding
                for i, dimension_size in reversed(list(enumerate(tensor.size())))
                for padding in [padding_lengths["dimension_" + str(i)] - dimension_size, 0]
            ]
        return torch.nn.functional.pad(tensor, pad, value=self.padding_value)
