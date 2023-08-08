import logging
import os

from typing import Tuple

import comet_ml

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction,
)
from peft import PeftConfig
from trl import SFTTrainer

from training.data import finqa
from training import models, constants, metrics


logger = logging.getLogger(__name__)


class FinQATrainingAPI:
    def __init__(
        self,
        root_dataset_dir: Path = constants.ROOT_DATASET_DIR_DEFAULT,
        model_id: str = constants.MODEL_ID_DEFAULT,
        training_arguments: TrainingArguments = constants.DEFAULT_TRAINING_ARGUMENTS,
        max_seq_length: int = 1024,
        debug: bool = False,
    ):
        self._root_dataset_dir = root_dataset_dir
        self._model_id = model_id
        self._training_arguments = training_arguments
        self._max_seq_length = max_seq_length
        self._debug = debug

        self._training_dataset, self._validation_dataset = self.load_data()
        self._model, self._tokenizer, self._peft_config = self.load_model()

    @property
    def name(self) -> str:
        return f"FinQA/{self._model_id}"

    def load_data(self) -> Tuple[Dataset, Dataset]:
        logger.info(f"Loading FinQA datasets from {self._root_dataset_dir=}")
        training_dataset = finqa.FinQADataset(
            data_path=self._root_dataset_dir / "train.json",
            scope=constants.Scope.TRAINING,
        ).to_huggingface()
        validation_dataset = finqa.FinQADataset(
            data_path=self._root_dataset_dir / "test.json",
            scope=constants.Scope.TRAINING,
        ).to_huggingface()

        if self._debug is True:
            logger.info("Debug mode enabled. Truncating datasets...")

            training_dataset = training_dataset.select(range(12))
            validation_dataset = validation_dataset.select(range(6))

        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        logger.info(f"Loading model using {self._model_id=}")
        model, tokenizer, peft_config = models.build_qlora_model(
            model_id=self._model_id, gradient_checkpointing=True
        )

        return model, tokenizer, peft_config

    def train(self) -> SFTTrainer:
        logger.info("Training model...")

        # TODO: Handle this error: "Token indices sequence length is longer than the specified maximum sequence length
        # for this model (2302 > 2048). Running this sequence through the model will result in indexing errors"
        trainer = SFTTrainer(
            model=self._model,
            train_dataset=self._training_dataset,
            eval_dataset=self._validation_dataset,
            peft_config=self._peft_config,
            dataset_text_field="text",
            max_seq_length=self._max_seq_length,
            tokenizer=self._tokenizer,
            args=self._training_arguments,
            packing=True,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        best_model_checkpoint = trainer.state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(f"Logging best model from {best_model_checkpoint} to the model registry...")

            self.log_model(best_model_checkpoint)
        else:
            logger.warning("No best model checkpoint found. Skipping logging it to the model registry...")

        return trainer
    
    def log_model(self, checkpoint_dir: Path):
        checkpoint_dir = checkpoint_dir.resolve()

        assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
        
        experiment = comet_ml.Experiment()
        experiment.log_model(self.name, str(checkpoint_dir)
)

    def compute_metrics(self, eval_pred: EvalPrediction):
        return {"perplexity": metrics.compute_perplexity(eval_pred.predictions)}
