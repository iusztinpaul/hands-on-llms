from enum import Enum
from pathlib import Path

from transformers import TrainingArguments


class Scope(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "test"


# TODO: Use Hydra as a configuration management tool.
# TODO: Configure this path instead of hardcoding it.
# ROOT_DIR = Path("/workspace")
ROOT_DIR = Path("..")
# TODO: Fix this /dataset/dataset nested directory.
# ROOT_DATASET_DIR_DEFAULT = ROOT_DIR / "dataset" / "dataset"
ROOT_DATASET_DIR_DEFAULT = ROOT_DIR / "dataset"
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
