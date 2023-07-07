local stringToInt(s) = std.parseInt(std.extVar(s));
local stringToBool(s) =
  if std.extVar(s) == "true" then true
  else if std.extVar(s) == "false" then false
  else error "invalid boolean: " + std.manifestJson(std.extVar(s));

local misc_params = {
  "model_name": std.extVar("model_name"),
  "cuda_device": stringToInt("gpu"),
  "batch_size": stringToInt("batch_size"),
  "epochs": stringToInt("epochs"),
  "n_samples_in_prompt": stringToInt("n_samples_in_prompt"),
  "seed": stringToInt("seed"),
  "wandb_group": std.extVar("wandb_group"),
  "wandb_project": std.extVar("wandb_project"),
  "selection_method": std.extVar("selection_method"),
  "retriever": std.extVar("retriever"),
  "extra_eval_split": std.extVar("extra_eval_split"),
  "dataset":  std.extVar("dataset"),
};

local target_max_tokens = 400;
local batch_size = misc_params.batch_size;
local max_instances = null;
local max_epochs = misc_params.epochs;
local program_based = true;

{
    "numpy_seed": misc_params.seed,
    "pytorch_seed": misc_params.seed,
    "random_seed": misc_params.seed,
    "train_data_path": "",
    "validation_data_path": "",
    "dataset_reader": {
        "type": "base_dataset_reader",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": misc_params.model_name,
            "add_special_tokens": false
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_custom",
                "model_name": misc_params.model_name,
                "namespace": "tokens",
                "padding_on_right": false,
            }
        },
        "source_max_tokens": 512,
        "target_max_tokens": target_max_tokens,
        "read_loops": 1,
        "n_train_sample": max_instances,
        "n_test_sample": max_instances,
        "sample_random_seed": 0,
        "validation_split": null,
        "extra_eval_split": misc_params.extra_eval_split,
    },
    "model": {
        "type": "seq2seq_transformers",
        "model_name": misc_params.model_name,
        "beam_size": 5,
        "max_decoding_steps": target_max_tokens,
    },
    "data_loader": {
        "type": "meta_learn_multiprocess",
        "batch_size": batch_size,
        "shuffle": true,
        "selection_method": {
            "type": misc_params.selection_method,
            "n_samples_in_prompt": misc_params.n_samples_in_prompt,
            "random_seed": misc_params.seed,
            "program_based": program_based,
            "retriever": {
                "type": misc_params.retriever,
            }
        },
        "verbalizer": {
            "type": "utterance_program",
            "model_name": misc_params.model_name,
        }
    },
    "trainer": {
        "type": "gradient_descent_fixed",
        "use_amp": false,
        "num_gradient_accumulation_steps": 1,
        "num_epochs": max_epochs,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true,
        },
        "validation_metric": "+accuracy",
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
            "warmup_steps": 100,
            "end_learning_rate": 1e-6
        },
        "cuda_device": misc_params.cuda_device,
        "grad_norm": 1.0,
        "callbacks": [
            {
                "type": "wandb_best",
                "should_log_learning_rate": true,
                "should_log_parameter_statistics": false,
                "project": misc_params.wandb_project,
                "group": misc_params.wandb_group
            },
            "save_predictions"
        ],
        "checkpointer": {
            "save_completed_epochs": false,
            "keep_most_recent_by_count": 1,
        },
        "run_confidence_checks": false
    }
}