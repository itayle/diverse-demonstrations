import random
from collections import defaultdict
from typing import List, Tuple

from allennlp.data import Instance

from src.selection_methods.selection_method import SelectionMethod


@SelectionMethod.register("cover_source")
class CoverSource(SelectionMethod):
    _atom_to_instances_qids = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    _templates_to_instances = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    def __init__(self,
                 is_test: bool,
                 instances: List[Instance],
                 anonymize_targets: bool = True,
                 no_template_repeat: bool = True,
                 **kwargs):
        super().__init__(is_test, instances, **kwargs)

        self._field_for_target = "anonymized_target" if anonymize_targets else "target"

        self._atom_to_instances_qids[self._split].clear()

        self._no_template_repeat = no_template_repeat
        self._atom_field = "source_atoms"

        for i, instance in enumerate(instances):
            if not instance["metadata"].get(self._field_for_target):
                raise ValueError("Cannot use cover-source without (anonymized) target")
            self._templates_to_instances[self._split][instance["metadata"][self._field_for_target]].add(instance["metadata"]["ex_qid"])
            for atom in instance['metadata'][self._atom_field]:
                self._atom_to_instances_qids[self._split][atom].add(instance["metadata"]["ex_qid"])


    def _inner_in_context_chooser(self,
                                  in_context_instances,
                                  num_samples_left,
                                  seen_templates,
                                  instance):
        atoms_pool = list(instance["metadata"][self._atom_field])
        uncovered_atoms = set(atoms_pool)

        no_more_found = False
            
        for i in range(num_samples_left):
            no_more_found = False
            relevant_atoms = set()

            for atom in atoms_pool:                
                instances_qids = self._atom_to_instances_qids["train"][atom]
                if len(instances_qids) == 0:
                    continue

                if instances_qids.issubset(seen_templates):
                    continue
                relevant_atoms.add(atom)

            if not relevant_atoms:
                # we've added all templates containing the ls of this size of this instance. We can (and probably should) now add
                # examples that were not seen yet, even though their template has been seen, but this is not implemented
                # yet.
                no_more_found = True
                break

            possible_atoms = uncovered_atoms.intersection(relevant_atoms)

            if not possible_atoms:
                # no atoms are left - start again with initial list of atoms
                break

            # sort to make randomness reproducible
            picked_atom = self._random.choice(sorted(list(possible_atoms)))

            # take all examples with these ls_pool
            examples_of_atom = self._atom_to_instances_qids["train"][picked_atom] - seen_templates

            relevant_instances = self._retriever.retrieve_instances(
                instance, sorted(examples_of_atom),
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

            for curr_atom in picked_instance['metadata'][self._atom_field]:
                if curr_atom in uncovered_atoms:
                    uncovered_atoms.remove(curr_atom)

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
            # start again with initial list of atoms
            num_samples_left = self._inner_in_context_chooser(in_context_instances, num_samples_left, seen_templates, instance)
        return in_context_instances, instance
