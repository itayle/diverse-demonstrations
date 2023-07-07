import json
from allennlp.common import FromParams
from typing import List


class PhaseOneExtractor(FromParams):
    def __init__(self,
                 phase_one_model: str = None,
                 phase_one_model_split: str = None,
                 ):
        self._prediction_per_qid = {}
        outputs_file_name = "best_outputs.jsonl" if not phase_one_model_split else f"{phase_one_model_split}"
        if phase_one_model:
            if phase_one_model.startswith("/"):
                predictor_path = f"{phase_one_model}/{outputs_file_name}"
            else:
                atom_predictor_model = phase_one_model.replace('/', '-')
                predictor_path = f"../runs/{atom_predictor_model}/{outputs_file_name}"
            predictor_model_outputs = [json.loads(l) for l in open(predictor_path)]

            for prediction_row in predictor_model_outputs:
                beam = [prediction_row[f'beam_{i}'] for i in range(10) if f'beam_{i}' in prediction_row]
                assert len(beam) > 0

                self._prediction_per_qid[prediction_row['qid']] = beam

    def get_beam_predictions(self, ex_qid: str, beam_size: int = None) -> List:
        if beam_size:
            return self._prediction_per_qid[ex_qid][:beam_size]
        else:
            return self._prediction_per_qid[ex_qid]
