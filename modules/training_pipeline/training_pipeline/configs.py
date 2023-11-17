from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from transformers import TrainingArguments

from training_pipeline.data.utils import load_yaml


@dataclass
class TrainingConfig:
    """
    Training configuration class used to load and store the training configuration.

    Attributes:
    -----------
    training : TrainingArguments
        The training arguments used for training the model.
    model : Dict[str, Any]
        The dictionary containing the model configuration.
    """

    training: TrainingArguments
    model: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path, output_dir: Path) -> "TrainingConfig":
        """
        Load a configuration file from the given path.

        Parameters:
        -----------
        config_path : Path
            The path to the configuration file.
        output_dir : Path
            The path to the output directory.

        Returns:
        --------
        TrainingConfig
            The training configuration object.
        """

        config = load_yaml(config_path)

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

        Parameters:
        -----------
        training_config : dict
            The dictionary containing the training configuration.
        output_dir : Path
            The path to the output directory.

        Returns:
        --------
        TrainingArguments
            The training arguments object.
        """

        return TrainingArguments(
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            eval_accumulation_steps=training_config["eval_accumulation_steps"],
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
    A class representing the configuration for inference.

    Attributes:
    -----------
    model : Dict[str, Any]
        A dictionary containing the model configuration.
    peft_model : Dict[str, Any]
        A dictionary containing the PEFT model configuration.
    setup : Dict[str, Any]
        A dictionary containing the setup configuration.
    dataset : Dict[str, str]
        A dictionary containing the dataset configuration.
    """

    model: Dict[str, Any]
    peft_model: Dict[str, Any]
    setup: Dict[str, Any]
    dataset: Dict[str, str]

    @classmethod
    def from_yaml(cls, config_path: Path):
        """
        Load a configuration file from the given path.
        """

        config = load_yaml(config_path)

        return cls(**config)
