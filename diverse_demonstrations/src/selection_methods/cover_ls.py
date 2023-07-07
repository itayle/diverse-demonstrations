import random
from collections import defaultdict
from typing import List, Tuple

from allennlp.data import Instance

from src.selection_methods.selection_method import SelectionMethod


@SelectionMethod.register("cover_ls")
class CoverLS(SelectionMethod):
    _ls_to_instances_qids = defaultdict(lambda: defaultdict(set))
    _templates_to_instances = defaultdict(lambda: defaultdict(set))

    def __init__(self,
                 is_test: bool,
                 instances: List[Instance],
                 anonymize_targets: bool = True,
                 no_template_repeat: bool = True,
                 limit_training_ls_size: float = None,
                 limit_eval_ls_size: float = None,
                 **kwargs):
        super().__init__(is_test, instances, **kwargs)

        self._field_for_target = "anonymized_target" if anonymize_targets else "target"

        self._ls_to_instances_qids[self._split].clear()

        self._no_template_repeat = no_template_repeat
        self._limit_training_ls_size = limit_training_ls_size
        self._limit_eval_ls_size = limit_eval_ls_size
        self._ls_field = "ls_by_size_flat"
        self._ls_field_by_size = "ls_by_size"

        for i, instance in enumerate(instances):
            if not instance["metadata"].get(self._field_for_target):
                raise ValueError("Cannot use cover-ls without (anonymized) target")
            self._templates_to_instances[self._split][instance["metadata"][self._field_for_target]].add(instance["metadata"]["ex_qid"])
            for ls in instance['metadata'][self._ls_field]:
                self._ls_to_instances_qids[self._split][ls].add(instance["metadata"]["ex_qid"])

    def _inner_in_context_chooser(self,
                                  in_context_instances,
                                  num_samples_left,
                                  seen_templates,
                                  instance):
        entire_ls_pool = list(instance["metadata"][self._ls_field])
        uncovered_ls = set(entire_ls_pool)

        ls_pool_by_size = [(ls_size, ls_pool) for ls_size, ls_pool in enumerate(instance["metadata"][self._ls_field_by_size])
                           if ls_pool]

        if self._limit_training_ls_size and not self._is_test:
            ls_pool_by_size = [(ls_size, ls_pool) for ls_size, ls_pool in ls_pool_by_size if ls_size < self._limit_training_ls_size]
        if self._limit_eval_ls_size and self._is_test:
            ls_pool_by_size = [(ls_size, ls_pool) for ls_size, ls_pool in ls_pool_by_size if ls_size < self._limit_eval_ls_size]
        if not ls_pool_by_size:
            raise NotImplementedError

        no_more_found = False
        
        for ls_size, ls_pool in reversed(ls_pool_by_size):
            # pick relevant ls_pool: those that should not be skipped, and that we have examples for
            
            for i in range(num_samples_left):
                no_more_found = False
                relevant_ls = set()

                for ls in ls_pool:                    
                    instances_qids = self._ls_to_instances_qids["train"][ls]
                    if len(instances_qids) == 0:
                        continue

                    if instances_qids.issubset(seen_templates):
                        continue
                    relevant_ls.add(ls)

                if not relevant_ls:
                    # we've added all templates containing the ls of this size of this instance. We can (and probably should) now add
                    # examples that were not seen yet, even though their template has been seen, but this is not implemented
                    # yet.
                    no_more_found = True
                    break

                possible_ls = uncovered_ls.intersection(relevant_ls)

                if not possible_ls:
                    break  # move to smaller ls_size

                # sort to make randomness reproducible
                picked_ls = self._random.choice(sorted(list(possible_ls)))

                # take all examples with these ls_pool
                examples_of_ls = self._ls_to_instances_qids["train"][picked_ls] - seen_templates

                relevant_instances = self._retriever.retrieve_instances(
                    instance, sorted(examples_of_ls),
                    n_samples=1
                )

                picked_instance_qid = self._random.choice(relevant_instances)
                picked_instance = self._instance_per_qid["train"][picked_instance_qid]

                if self._no_template_repeat:
                    # all instances with similar templates will not be seen again
                    seen_templates.update(self._templates_to_instances["train"][picked_instance['metadata'][self._field_for_target]])
                else:
                    # only this instance will not be seen again
                    seen_templates.add(picked_instance_qid)

                for curr_ls in picked_instance['metadata'][self._ls_field]:
                    if curr_ls in uncovered_ls:
                        uncovered_ls.remove(curr_ls)

                in_context_instances.append(picked_instance)
                num_samples_left -= 1
        if no_more_found:
            return -1
        return num_samples_left

    def _get_in_context_instances(self, instance: Instance) -> Tuple[List[Instance], Instance]:
        instance_qid = instance["metadata"]["ex_qid"]
        in_context_instances = []
        num_samples_left = self._n_samples_in_prompt
        seen_templates = {instance_qid}

        if not self._is_test:
            seen_templates.update(self._templates_to_instances["train"][instance['metadata'][self._field_for_target]])

        while num_samples_left > 0:
            # start again with initial list of ls
            num_samples_left = self._inner_in_context_chooser(in_context_instances, num_samples_left, seen_templates, instance)
        return in_context_instances, instance
