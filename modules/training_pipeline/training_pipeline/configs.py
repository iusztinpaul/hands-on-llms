from dataclasses import dataclass
from typing import Any, Dict
import yaml

from pathlib import Path

from transformers import TrainingArguments


@dataclass
class TrainingConfig:
    """
    Training configuration class used to load and store the training configuration.
    """

    training: TrainingArguments
    model: Dict[str, Any]
    setup: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path, output_dir: Path):
        """
        Load a configuration file from the given path.
        """

        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        config["training"] = cls._dict_to_training_arguments(
            training_config=config["training"], output_dir=output_dir
        )

        return cls(**config)

    @classmethod
    def _dict_to_training_arguments(
        cls, training_config: dict, output_dir: Path
    ) -> TrainingArguments:
        """
        Build a TrainingArguments object from a configuration dictionary.
        """

        return TrainingArguments(
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            optim=training_config["optim"],
            save_steps=training_config["save_steps"],
            logging_steps=training_config["logging_steps"],
            learning_rate=training_config["learning_rate"],
            fp16=training_config["fp16"],
            max_grad_norm=training_config["max_grad_norm"],
            num_train_epochs=training_config["num_train_epochs"],
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            evaluation_strategy=training_config["evaluation_strategy"],
            eval_steps=training_config["eval_steps"],
            report_to=training_config["report_to"],
            seed=training_config["seed"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
        )

@dataclass
class InferenceConfig:
    """
    Inference configuration class used to load and store the inference configuration.
    """

    model: Dict[str, Any]
    peft_model: Dict[str, Any]
    setup: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path):
        """
        Load a configuration file from the given path.
        """

        # TODO: Load peft_model from model registry. 

        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        return cls(**config)
