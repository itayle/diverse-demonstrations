import logging
import re
from typing import Iterator, TypeVar

from allennlp.common import Lazy
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.data_loaders.data_loader import DataLoader
from allennlp.data.instance import Instance
from overrides import overrides

from src.selection_methods.selection_method import SelectionMethod
from src.verbalizers.verbalizer import Verbalizer

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@DataLoader.register("meta_learn_multiprocess")
class MetaLearnMultiProcessDataLoader(MultiProcessDataLoader):

    def __init__(self,
                 selection_method: Lazy[SelectionMethod],
                 verbalizer: Lazy[Verbalizer],
                 **kwargs):
        data_path = kwargs['data_path']
        self._is_test = '@' in data_path
        if self._is_test:
            kwargs['shuffle'] = False
            print("Manually setting validation_data_loader.shuffle = False")
       
        dataset = re.search(r"datasets/(.*)/", data_path)
        if dataset:
            self.dataset = dataset.groups()[0]
        else:
            raise Exception("cant extract dataset name from data_path")
        print(f"----Creating {'test' if self._is_test else 'train'} data loader for dataset {self.dataset}----\n")
        super().__init__(**kwargs)
        print(f"Actual number of loaded instances: {len(self._instances)}")
        if self.max_instances_in_memory is not None:
            raise NotImplementedError()


        self.selection_method = selection_method.constructor(
            is_test=self._is_test,
            instances=list(self._instances),
            dataset=self.dataset,
        )
        self.verbalizer = verbalizer.constructor(
            is_test=self._is_test,
            dataset=self.dataset,
        )

    @overrides
    def iter_instances(self) -> Iterator[Instance]:
        if not self._instances or not self._vocab:
            yield from super().iter_instances()
        else:
            instance_iterator = (self._index_instance(instance) for instance in super().iter_instances())
            for test_instance in instance_iterator:
                in_context_instances, test_instance = self.selection_method.get_in_context_instances(test_instance)
                prompt_instance = self.verbalizer.construct_prompt(in_context_instances, test_instance)
                yield prompt_instance
