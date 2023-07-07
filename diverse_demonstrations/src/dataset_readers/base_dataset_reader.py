import json
import logging
import itertools
from random import Random
from typing import Dict, Optional, Iterable

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from overrides import overrides

from src.local_structures_extractors.ngram_extractor_ast import NgramsExtractorAST
from src.utils.atoms_utils import get_atoms, tokenize_lf
from src.utils.phase_one_extract import PhaseOneExtractor
logger = logging.getLogger(__name__)

GOLD_TARGET_TYPE_PREFIX = 'gold_'
PREDICTED_TARGET_TYPE_PREFIX = ''

@DatasetReader.register("base_dataset_reader")
class BaseDatasetReader(DatasetReader):
    instances_per_split = {}

    def __init__(
        self,
        phase_one_model: str = None,
        phase_one_model_split: str = None,
        phase_one_beam_size: int = 1,
        split_path: str = None,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        read_loops: int = 1,
        n_train_sample: int = None,
        n_test_sample: int = None,
        sample_random_seed: int = 0,
        instance_source: str = 'source',
        ls_size: int = 2,
        ls_siblings: bool = False,
        validation_split: str = None,
        extra_eval_split: str = None,
        anonymize_target: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._split_path = split_path
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens

        self._read_loops = read_loops

        self._n_train_sample = n_train_sample
        self._n_test_sample = n_test_sample

        self._random = Random(sample_random_seed)

        self._instance_source = instance_source

        self._ls_size = ls_size
        self._ls_siblings = ls_siblings
        self._ls_extractor = NgramsExtractorAST({
            'n': self._ls_size,
            'add_adjacent_sibling': self._ls_siblings,
            'add_parent_child': True,
            'add_prefix': True,
        })

        self._validation_split = validation_split
        self._extra_eval_split = extra_eval_split

        self._anonymize_target = anonymize_target
        self._phase_one_extractor = PhaseOneExtractor(phase_one_model=phase_one_model, phase_one_model_split=phase_one_model_split) if phase_one_model else None
        self._phase_one_beam_size = phase_one_beam_size

    def get_examples(self, file_path, is_train: bool):
        all_examples = [json.loads(s) for s in open(file_path, 'rt')]
        selected_examples = all_examples

        if self._split_path:
            split_info = json.load(open(self._split_path, 'rt'))
            train_set_ids = split_info.get('train')
            extra_train_eval_ids = None
            train_ids = set(train_set_ids)
            test_ids = set(split_info.get(self._validation_split))

            if is_train:
                selected_examples = [ex for ex in all_examples if ex['qid'] in train_ids]
            else:
                selected_examples = [ex for ex in all_examples if ex['qid'] in test_ids]

                if self._extra_eval_split:
                    extra_eval_ids = set(split_info.get(self._extra_eval_split))
                    extra_examples = [ex for ex in all_examples if ex['qid'] in extra_eval_ids]
                    for ex in extra_examples:
                        ex["_extra_eval_split"] = self._extra_eval_split
                    selected_examples += extra_examples

                if extra_train_eval_ids is not None:
                    extra_examples = [ex for ex in all_examples if ex['qid'] in extra_train_eval_ids]
                    for ex in extra_examples:
                        ex["_extra_eval_split"] = f"train_set_split_{self._train_set_split_index + 1}"
                    selected_examples += extra_examples

        assert len(selected_examples) > 0
        return selected_examples

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        is_test = file_path.startswith('@')
        if is_test:
            file_path = file_path[1:]
        is_train = not is_test

        all_examples = self.get_examples(file_path, is_train)
        original_size = len(all_examples)

        if self._n_train_sample and is_train and len(all_examples) > self._n_train_sample:
            all_examples = self._random.sample(all_examples, self._n_train_sample)
        elif self._n_test_sample and not is_train and len(all_examples) > self._n_test_sample:
            # we want test sampling to be the same no matter the training seed - so we use a separate random object
            sampling_random = Random(0)
            all_examples = sampling_random.sample(all_examples, self._n_test_sample)

        self.instances_per_split['train' if is_train else 'test'] = []

        print(f"\nTrying to load {len(all_examples)} {'train' if is_train else 'test'} examples "
              f"(original size was {original_size})")
        read_loops = self._read_loops if not is_test else 1
        for _ in range(read_loops):
            for line in all_examples:
                instance = self.text_to_instance(ex=line, is_test=is_test)
                if instance:
                    self.instances_per_split['train' if is_train else 'test'].append(instance)
                    yield instance


    @overrides
    def text_to_instance(self, ex: Dict, is_test: bool) -> Optional[Instance]:
        source = ex[self._instance_source] + " "
        tokenized_source = self._source_tokenizer.tokenize(source)

        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            raise Exception(f"Source is of length {len(tokenized_source)}, which is longer than source_max_tokens"
                            f" of {self._source_max_tokens}")
        source_field = TextField(tokenized_source, self._source_token_indexers)

        metadata = {
            'ex_qid': ex['qid'],
            'source': source,
            'source_atoms': source.lower().split()
        }

        fields = {
            "source_tokens": source_field,
        }

        if ex.get('target'):
            target = metadata['target'] = ex['target']
            anonymized_target = metadata['anonymized_target'] = ex['anonymized'] if not self._anonymize_target else target

            tokenized_target = self._target_tokenizer.tokenize(target)
            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields["target_tokens"] = target_field
            
            if self._phase_one_extractor and is_test:
                beam = beam_anonymized = self._phase_one_extractor.get_beam_predictions(ex['qid'], self._phase_one_beam_size)
                metadata['beam'] = self._phase_one_extractor.get_beam_predictions(ex['qid'], None)
            else:
                beam = [target]
                beam_anonymized = [anonymized_target]
            for target_type_prefix in [GOLD_TARGET_TYPE_PREFIX, PREDICTED_TARGET_TYPE_PREFIX]:
                target_var = [target] if target_type_prefix == GOLD_TARGET_TYPE_PREFIX else beam
                target_var_anonymized = [anonymized_target] if target_type_prefix == GOLD_TARGET_TYPE_PREFIX else beam_anonymized

                metadata[target_type_prefix + 'atoms'] = self._get_atoms(target_var)
                metadata[target_type_prefix + 'atoms_count'] = len(metadata[target_type_prefix + 'atoms'])
                metadata[target_type_prefix + 'atoms_anonymized'] = self._get_atoms(target_var_anonymized)
                metadata[target_type_prefix + 'local_structures'] = self._get_local_structures(target_var)
                metadata[target_type_prefix + 'local_structures_anonymized'] = self._get_local_structures(target_var_anonymized)

                ls_by_size = [[] for _ in range(self._ls_size)]
                ls_by_size[0] = list(metadata[target_type_prefix + 'atoms_anonymized'])
                for ls in metadata[target_type_prefix + 'local_structures_anonymized']:
                    ls_by_size[len(ls)-1].append(str(ls))
                metadata[target_type_prefix + 'ls_by_size'] = ls_by_size
                # unites atoms and local structures
                metadata[target_type_prefix + 'ls_by_size_flat'] = list(itertools.chain(*ls_by_size))

            tags = ex.pop('tags', [])
            metadata['tags'] = tags
        metadata["extra_eval_split"] = ex.get("_extra_eval_split", 0)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _get_atoms(self, beam):
        atoms = set()
        for pred in beam:
            pred_atoms = get_atoms(target=pred)
            pred_atoms = [a.replace("(", "").replace(")", "").replace("[", "").replace("]", "") for a in pred_atoms]
            atoms.update(pred_atoms)
        
        for atom in atoms:
            NgramsExtractorAST.set_inner_ls(tuple([atom]))
                
        return sorted(atoms)

    def _get_local_structures(self, beam):
        local_structures = set()
        for pred in beam:
            # fix a common dot issue
            if pred[-1] == ".":
                pred = pred[:-1]

            # fix a common parentheses issue
            missing_closing_parentheses = pred.count("(") - pred.count(")")
            if missing_closing_parentheses > 0:
                pred = pred + (")" * missing_closing_parentheses)
            elif missing_closing_parentheses < 0:
                while missing_closing_parentheses < 0:
                    if pred.strip()[-1] == ")":
                        pred = pred.strip()[:-1]
                        missing_closing_parentheses += 1
                    else:
                        break
            try:
                local_structures.update(self._ls_extractor.get_ngrams_from_target(tokenize_lf(pred)))
            except Exception as e:
                print(f"\nException occurred during ngram extraction of example with target: {pred}")
        return sorted(local_structures)