from typing import Dict, Optional, Any

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer.util import IntT

from src.models.base_transformers import Transformers
from src.models.seq2seq.t5_wrapper import T5Wrapper


@Model.register("seq2seq_transformers")
class Seq2SeqTransformers(Transformers):
    def __init__(self, 
                 vocab: Vocabulary,
                 sampler_type: str = "deterministic",
                 final_sequence_scorer_type: str = "sequence-log-prob",
                 **kwargs) -> None:
        super().__init__(vocab, **kwargs)

        self.model = T5Wrapper(vocab, 
                               self._model_name, 
                               max_decoding_steps=self._max_decoding_steps,
                               beam_size=self._beam_size, 
                               sampler_type=sampler_type,
                               final_sequence_scorer_type=final_sequence_scorer_type)
    def forward(  # type: ignore
            self,
            source_tokens: TextFieldTensors,
            target_tokens: Optional[TextFieldTensors] = None,
            metadata: Dict = None
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

        if target_tokens is not None:
            labels, decoder_attention_mask = (
                target_tokens["tokens"]["token_ids"],
                target_tokens["tokens"]["mask"],
            )
        elif self.training:
            raise ValueError("'target_tokens' required during training")

        output_dict = self.model(source_tokens, target_tokens)
        output_dict["qid"] = [md['ex_qid'] for md in metadata]
        output_dict["source_text"] = [md['source'] for md in metadata]
        output_dict["gold_text"] = [md['target'] for md in metadata]
        output_dict["meta_train_sources"] = [md['meta_train_sources'] for md in metadata]
        output_dict["meta_train_targets"] = [md['meta_train_targets'] for md in metadata]

        output_dict["n_actual_prompt_samples"] = [len(md['prompts']) for md in metadata]
        output_dict["input_length"] = (input_ids != 0).sum(axis=1).tolist()

        if not self.training:
            if labels is not None:
                output_dict["predicted_text"] = self.make_output_human_readable(output_dict['predictions'])

                if "beam_predictions" in output_dict:
                    for i in range(self._beam_size):
                        beam_predictions = output_dict["beam_predictions"][:, i]
                        output_dict[f'beam_{i}'] = self.make_output_human_readable(beam_predictions)
                    del output_dict["beam_predictions"]

                output_dict["predicted_target"] = []
                # wandb lists
                output_dict["gold vs predicted"] = list(
                    map(list, zip(output_dict["gold_text"], output_dict["predicted_text"])))
                gold_tokenized_text = self.make_output_human_readable(labels)

                for pred, gold, md, *beam in zip(output_dict['predicted_text'], gold_tokenized_text, metadata):
                    self.update_metrics(pred=pred, gold=gold, metadata=md, output_dict=output_dict)

        return output_dict

    def make_output_human_readable(self, sequence) -> Dict[str, Any]:
        decoded_batch = self.tokenizer.batch_decode(
            sequence, skip_special_tokens=True
        )
        return decoded_batch
