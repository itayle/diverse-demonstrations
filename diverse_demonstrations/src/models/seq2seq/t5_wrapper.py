from typing import Optional, Dict

import torch
from allennlp.common import Lazy, Params
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules.transformer.util import IntT, BoolT
from allennlp.nn.beam_search import BeamSearch

from src.models.seq2seq.t5_module import T5, T5Output


class T5Wrapper(Model):
    def __init__(self, 
                 vocab: Vocabulary, 
                 model_name: str, 
                 beam_size: int = 3, 
                 max_decoding_steps: int = 100, 
                 sampler_type: str = "deterministic",
                 final_sequence_scorer_type: str = "sequence-log-prob",
                 **kwargs) -> None:
        super().__init__(vocab, **kwargs)
        self._beam_size = beam_size
        self._model_name = model_name
        self._sampler_type = sampler_type
        sampler_dict = {
            "type": self._sampler_type,
        }
        if self._sampler_type == "top-p_with_replacement":
            sampler_dict["type"] = "top-p"
            sampler_dict["with_replacement"] = True            
        self._final_sequence_scorer_type = final_sequence_scorer_type
        scorer_dict = {
            "type": self._final_sequence_scorer_type,
        }
        beam_search = Lazy(
            BeamSearch,
            params=Params(
                {
                    "beam_size": beam_size,
                    "max_steps": max_decoding_steps,
                    "sampler": sampler_dict,
                    "final_sequence_scorer": scorer_dict
                }
            ),
        )
        self.model = T5.from_pretrained_module(model_name, beam_search=beam_search)

    def forward(  # type: ignore
            self,
            source_tokens: TextFieldTensors,
            target_tokens: Optional[TextFieldTensors] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of T5.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key/namespace.

        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are also stored under the `tokens` key/namespace.
            If no target tokens are given during training / validation, the source tokens are shifted
            to the right by 1.

        # Returns

        `Dict[str, torch.Tensor]`
            Contains the `loss` when `target_tokens` is provided.
            And during prediction, includes `predictions` and `predicted_log_probs` from beam search.

        """
        input_ids, attention_mask = (
            source_tokens["tokens"]["token_ids"],
            source_tokens["tokens"]["mask"],
        )
        labels: Optional[IntT] = None
        decoder_attention_mask: Optional[BoolT] = None
        if target_tokens is not None:
            labels, decoder_attention_mask = (
                target_tokens["tokens"]["token_ids"],
                target_tokens["tokens"]["mask"],
            )
        elif self.training:
            raise ValueError("'target_tokens' required during training")

        output: T5Output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        output_dict: Dict[str, torch.Tensor] = {}

        if self.training:
            assert output.loss is not None
            output_dict["loss"] = output.loss
        else:
            # Shape: (batch_size, beam_size, num_tokens)
            assert output.predictions is not None
            # Shape: (batch_size, beam_size)
            assert output.predicted_log_probs is not None
            # Shape: (batch_size, num_tokens)
            output_dict["predictions"] = output.predictions[:, 0, :]
            # Shape (predicted_log_probs):   (batch_size, beam_size)
            for i in range(self._beam_size):
                output_dict[f"predicted_log_probs_beam_{i}"] = output.predicted_log_probs[:, i]
            output_dict["predicted_log_probs"] = output.predicted_log_probs[:, 0]

            output_dict["beam_predictions"] = output.predictions

            if labels is not None:
                assert output.loss is not None
                output_dict["loss"] = output.loss

        return output_dict
