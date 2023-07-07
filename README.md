# Diverse Demonstrations Improve In-context Compositional Generalization

Accepted at ACL 2023.

**Link to the Paper**: [arXiv](https://arxiv.org/abs/2212.06800)
## Authors

- Itay Levy†
- Ben Bogin†
- Jonathan Berant

† - Equal contribution.
## Abstract

In-context learning has shown great success in i.i.d semantic parsing splits, where the training and test sets are drawn from the same distribution. In this setup, models are typically prompted with demonstrations that are similar to the input utterance. However, in the setup of compositional generalization, where models are tested on outputs with structures that are absent from the training set, selecting similar demonstrations is insufficient, as often no example will be similar enough to the input. In this work, we propose a method to select diverse demonstrations that aims to collectively cover all of the structures required in the output program, in order to encourage the model to generalize to new structures from these demonstrations. We empirically show that combining diverse demonstrations with in-context learning substantially improves performance across three compositional generalization semantic parsing datasets in the pure in-context learning setup and when combined with finetuning.


## Table of Contents

1. [Installation](#installation)
2. [Phase 1 Training](#phase-1-training)
3. [Saving Predictions](#saving-predictions)
4. [NOFT Experiment](#noft-experiment)
5. [FT Experiment](#ft-experiment)

## Installation

### Prerequisites

This project requires the following:

1. python >= 3.7.11
2. torch >= 1.8.2
3. allennlp >= 2.9.0

### Getting Started

To install the dependencies for this project, follow the steps below:

1. Clone this repository

2. (Optional) Create a virtual environment:


3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
4. Navigate to the datasets directory and decompress the dataset:

    ```bash
    cd datasets
    tar -xvzf datasets.tgz
    ```


## Phase 1 Training

USe the following command to train the phase 1 model on the standard split of the GeoQuery dataset:

    
    python train.py --dataset geo880_v2 --split-name standard --selection_method topk --n_samples_in_prompt 0 --epochs 250 --retriever random --eval-retriever random --validation_split test --batch_size 8 --validate_every 1

## Saving Predictions

To evaluate the model and save predctions to a file to be used later, use the following command:

   
    python evaluate_FT.py evaluate <run_dir> ../datasets/geo880_v2/all.jsonl --batch-size 8 --cuda-device 0 --predictions-output-file <phase_one_model/phase_one_model_split> --output-file <metrics_file.json> --overrides {'dataset_reader.validation_split':'test','model.max_decoding_steps':700,'model.sampler_type':'deterministic','model.final_sequence_scorer_type':'length-normalized-sequence-log-prob','dataset_reader.extra_eval_split':null,'model.extra_valid_accuracy_names':null}


`phase_one_model` determines the directory where the predictions are saved and `phase_one_model_split` determines the file name for the predictions (we recommend using a jsonl file).
In the following steps you will need to use those two arguments to load the predictions. 


## NOFT Experiment

To run NOFT experiment using OpenAI API, set the desired arguments in the `evaluate_NOFT.py` file and run the following command:


    python evaluate_NOFT.py


## FT Experiment
To run FT experiment using T5 model, run the following command:


    python train.py --dataset geo880_v2 --split-name standard --selection_method topk --n_samples_in_prompt 3 --epochs 250 --retriever topk --eval-retriever topk --validation_split test --batch_size 8 --validate_every 1  --overrides {'dataset_reader.phase_one_model':<phase_one_model>,'dataset_reader.phase_one_model_split':<phase_one_model_split>,'dataset_reader.phase_one_beam_size':<n_beams>}



