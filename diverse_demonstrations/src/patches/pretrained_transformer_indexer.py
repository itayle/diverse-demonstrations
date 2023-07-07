import logging
from typing import Dict

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer_custom")
class PretrainedTransformerIndexerCustom(PretrainedTransformerIndexer):
    """
    Same as `pretrained_transformer` except we can control the padding here
    """  # noqa: E501

    def __init__(
            self,
            padding_on_right: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.padding_on_right = padding_on_right

    def as_padded_tensor_dict(
            self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for key, val in tokens.items():
            if key == "type_ids":
                padding_value = 0
                mktensor = torch.LongTensor
            elif key == "mask" or key == "wordpiece_mask":
                padding_value = False
                mktensor = torch.BoolTensor
            elif len(val) > 0 and isinstance(val[0], bool):
                padding_value = False
                mktensor = torch.BoolTensor
            else:
                padding_value = self._tokenizer.pad_token_id
                if padding_value is None:
                    padding_value = (
                        0  # Some tokenizers don't have padding tokens and rely on the mask only.
                    )
                mktensor = torch.LongTensor

            tensor = mktensor(
                pad_sequence_to_length(
                    val, padding_lengths[key], default_value=lambda: padding_value, padding_on_right=self.padding_on_right
                )
            )

            tensor_dict[key] = tensor
        return tensor_dict
