from enum import Enum
from pathlib import Path

import yaml

from transformers import TrainingArguments


class Scope(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "test"


# TODO: Use Hydra as a configuration management tool.
# TODO: Configure this path instead of hardcoding it.
ROOT_DIR = Path("/workspace")
# ROOT_DIR = Path("..") / ".."
# TODO: Fix this /dataset/dataset nested directory.
ROOT_DATASET_DIR_DEFAULT = ROOT_DIR / "dataset" / "dataset"
# ROOT_DATASET_DIR_DEFAULT = ROOT_DIR / "dataset"
MODEL_ID_DEFAULT = "tiiuae/falcon-7b-instruct"

RESULT_DIR_DEFAULT = ROOT_DIR / "results"
LOGGING_DIR_DEFAULT = ROOT_DIR / "logs"
DEFAULT_TRAINING_ARGUMENTS = TrainingArguments(
    output_dir=str(RESULT_DIR_DEFAULT),
    logging_dir=str(LOGGING_DIR_DEFAULT),
    per_device_train_batch_size=1,  # increase this value if you have more VRAM
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,  # increase this value if you have more VRAM
    optim="paged_adamw_32bit",  # This parameter activate QLoRa's pagination # TODO: Should we use paged_adamw_8bit instead?
    save_steps=1,
    logging_steps=1,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=1,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    evaluation_strategy="steps",
    eval_steps=1,
    report_to="comet_ml",
    seed=42,
    load_best_model_at_end=True, # By default it will use the 'loss' to find the best checkpoint. Leverage the 'metric_for_best_model' paramater if you want to use a different metric. 
)


def build_training_arguments(config: dict, output_dir: Path) -> TrainingArguments:
    """
    Build a TrainingArguments object from a configuration dictionary.
    """

    training_arguments = config["training"]

    return TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
        per_device_train_batch_size=training_arguments["per_device_train_batch_size"],
        gradient_accumulation_steps=training_arguments["gradient_accumulation_steps"],
        per_device_eval_batch_size=training_arguments["per_device_eval_batch_size"],
        optim=training_arguments["optim"],
        save_steps=training_arguments["save_steps"],
        logging_steps=training_arguments["logging_steps"],
        learning_rate=training_arguments["learning_rate"],
        fp16=training_arguments["fp16"],
        max_grad_norm=training_arguments["max_grad_norm"],
        num_train_epochs=training_arguments["num_train_epochs"],
        warmup_ratio=training_arguments["warmup_ratio"],
        lr_scheduler_type=training_arguments["lr_scheduler_type"],
        evaluation_strategy=training_arguments["evaluation_strategy"],
        eval_steps=training_arguments["eval_steps"],
        report_to=training_arguments["report_to"],
        seed=training_arguments["seed"],
        load_best_model_at_end=training_arguments["load_best_model_at_end"],
    )


def load_config(config_path: Path) -> dict:
    """
    Load a configuration file from the given path.
    """

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config
