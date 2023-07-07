from abc import abstractmethod

from typing import Iterator, TypeVar, List, Dict
from collections import defaultdict
import itertools

import numpy as np
from allennlp.common import cached_transformers, Lazy
from allennlp.data import Token
from allennlp.data.fields import MetadataField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from overrides import overrides

from src.patches.tensor_field import TensorField
from allennlp.common import Registrable
from allennlp.data import Instance, Token
import transformers


class Verbalizer(Registrable):
    default_implementation = "utterance_program"

    def __init__(self,
                 is_test: bool,
                 model_name: str,
                 max_prompt_size: int = None,
                 ignore_meta_train_source: bool = True,
                 ):
        self._is_test = is_test
        self._tokenizer = cached_transformers.get_tokenizer(model_name)
        self._model_max_length = self._tokenizer.model_max_length
        if max_prompt_size:
            self._model_max_length = min(self._tokenizer.model_max_length, max_prompt_size)
        # these were arbitrarily selected for now - should check MetaICL paper to see if there are any specific choices
        self._pre_atom_separator_str = " atom: "
        self._pre_utterance_separator_str = "utterance: "
        self._pre_program_separator_str = " program: "
        self._pre_atom_separator = self._tokenize_str(self._pre_atom_separator_str)
        self._pre_utterance_separator = self._tokenize_str(self._pre_utterance_separator_str)
        self._pre_program_separator = self._tokenize_str(self._pre_program_separator_str)
        self._bos_tokens = [Token(self._tokenizer.bos_token)] if self._tokenizer.bos_token is not None else []
        self._eos_tokens = [Token(self._tokenizer.eos_token)]
        self._ignore_meta_train_source = ignore_meta_train_source

    @abstractmethod
    def _verbalize_prompt(self,
                          meta_train_instances: List[Instance],
                          max_length: int):
        raise NotImplementedError

    def construct_prompt(self, in_context_instances: List[Instance], test_instance: Instance) -> Instance:
        """
        Creates a prompt instance from a list of in-context instances. The output instance contains only a `tokens`
        field (since the input to an LM should not separate source and target tokens)
        """
        merged_metadata = {}
        merged_metadata.update(test_instance['metadata'].metadata)

        # list of all examples, including both meta-train and meta-test examples
        all_examples = in_context_instances + [test_instance]

        token_indexers = test_instance['source_tokens'].token_indexers
        return self._create_instance_with_separated_input_output(merged_metadata, all_examples, token_indexers)

    def _create_metadata_for_tokens(self, tokens: List[Token], name: str = ""):
        metadata_dict = {name + "_tokens_length": len(tokens),
                         name + "_tokenizer_str": self._tokenizer.convert_tokens_to_string(
                             [token.text for token in tokens])}
        return metadata_dict

    def _tokenize_str(self, str_for_tokenization):
        return [Token(t) for t in self._tokenizer.tokenize(str_for_tokenization)]

    def _get_meta_test_source_tokens(self, meta_test_example):
        return self._pre_utterance_separator + meta_test_example['source_tokens'].tokens

    def _get_meta_test_target_tokens(self, meta_test_example):
        return self._pre_program_separator + meta_test_example['target_tokens'].tokens


    def _create_instance_with_separated_input_output(self, metadata, all_instances: List[Instance], token_indexers):
        meta_train_examples = all_instances[:-1]
        meta_test_example = all_instances[-1]

        meta_test_tokens = self._get_meta_test_source_tokens(meta_test_example)
        metadata.update(self._create_metadata_for_tokens(meta_test_tokens, "meta_test_source"))

        # maximum length for the meta-train tokens. Add 2 for BOS/EOS
        maximum_meta_train_length = self._model_max_length - len(meta_test_tokens) - 2

        meta_train_tokens, meta_train_metadata = self._verbalize_prompt(
            meta_train_examples, maximum_meta_train_length
        )

        prompt_tokens = (self._bos_tokens + meta_test_tokens + meta_train_tokens + self._eos_tokens)
        metadata.update(meta_train_metadata)

        # this is the default behaviour (padding added to the right), in contrast to the combined input/output version
        token_indexers['tokens'].padding_on_right = True

        # add test instance source to our prompt
        source_tokens = prompt_tokens
        source_tokens_field = TextField(tokens=source_tokens, token_indexers=token_indexers)
        source_tokens_field.index(vocab=None)
        metadata.update(self._create_metadata_for_tokens(source_tokens, "entire_source"))

        
        target_tokens = self._bos_tokens + meta_test_example['target_tokens'].tokens + self._eos_tokens

        target_tokens_field = TextField(tokens=target_tokens, token_indexers=token_indexers)
        target_tokens_field.index(vocab=None)
        metadata.update(self._create_metadata_for_tokens(target_tokens, "meta_test_target"))

        return Instance(fields={
            'source_tokens': source_tokens_field,
            'target_tokens': target_tokens_field,
            'metadata': MetadataField(metadata)
        })
