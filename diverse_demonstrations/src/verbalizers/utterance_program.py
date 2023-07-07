from typing import List, Dict

from allennlp.data import Instance

from src.verbalizers.verbalizer import Verbalizer


@Verbalizer.register("utterance_program")
class UtteranceProgramVerbalizer(Verbalizer):

    def _verbalize_prompt(self,
                          meta_train_instances: List[Instance],
                          max_length: int,
                         ):
        prompts_metadata = {}
        metadata = {"prompts": prompts_metadata}
        meta_train_tokens = []
        meta_train_sources = []
        meta_train_targets = []
        for i, instance in enumerate(meta_train_instances):
            candidate_source_tokens = self._pre_utterance_separator + instance['source_tokens'].tokens
            candidate_prompt_tokens = self._pre_program_separator + instance['target_tokens'].tokens
            if not self._ignore_meta_train_source:
                candidate_prompt_tokens = candidate_source_tokens + candidate_prompt_tokens

            if len(meta_train_tokens) + len(candidate_prompt_tokens) > max_length:
                break

            meta_train_tokens += candidate_prompt_tokens
            current_prompt_metadata = instance['metadata'].metadata.copy()
            current_prompt_metadata.update(self._create_metadata_for_tokens(candidate_prompt_tokens))
            prompts_metadata[f'prompt_{i}'] = current_prompt_metadata

            current_source_str = current_prompt_metadata['source'] if not self._ignore_meta_train_source else ""
            meta_train_sources.append(current_source_str)
            meta_train_targets.append(current_prompt_metadata['target'])

        metadata.update(self._create_metadata_for_tokens(meta_train_tokens, "meta_train"))
        metadata["meta_train_sources"] = meta_train_sources
        metadata["meta_train_targets"] = meta_train_targets
        return meta_train_tokens, metadata
