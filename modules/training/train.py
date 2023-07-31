import logging
import comet_ml
import os
import numpy as np

from dotenv import load_dotenv, find_dotenv

from pathlib import Path
from transformers import TrainingArguments, EvalPrediction
from trl import SFTTrainer

from training.data import finqa
from training import models, constants


logger = logging.getLogger(__name__)


ROOT_DATASET_DIR_DEFAULT = Path("../dataset")
MODEL_ID_DEFAULT = "tiiuae/falcon-7b-instruct"
RESULT_DIR_DEFAULT = Path("../results")
LOGGING_DIR_DEFAULT = Path("../logs")


# TODO: Move this to a class.
def train(
        root_dataset_dir: Path = ROOT_DATASET_DIR_DEFAULT,
        model_id: str = MODEL_ID_DEFAULT,
        result_dir: Path = RESULT_DIR_DEFAULT,
        logging_dir: Path = LOGGING_DIR_DEFAULT,
        debug: bool = False
        ):
    logger.info("Initializing...")
    initialize()

    logger.info("Loading datasets...")
    training_dataset = finqa.FinQADataset(
        data_path=root_dataset_dir / "train.json", scope = constants.Scope.TRAINING
    ).to_huggingface()
    validation_dataset = finqa.FinQADataset(
        data_path=root_dataset_dir / "test.json", scope=constants.Scope.TRAINING
    ).to_huggingface()

    logger.info("Loading model...")
    model, tokenizer, peft_config = models.build_qlora_model(model_id=model_id)

    if debug:
        logger.info("Debug mode enabled. Truncating datasets...")
        training_dataset = training_dataset.select(range(99))
        validation_dataset = validation_dataset.select(range(99))

    # TODO: Inject TrainingArguments as parameters.
    training_arguments = TrainingArguments(
        output_dir=str(result_dir),
        logging_dir=str(logging_dir),
        per_device_train_batch_size=1,  # increase this value if you have more VRAM
        gradient_accumulation_steps=3,
        per_device_eval_batch_size=1,  # increase this value if you have more VRAM
        optim="paged_adamw_32bit",  # This parameter activate QLoRa's pagination
        save_steps=40,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=9,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        evaluation_strategy="steps",
        eval_steps=3,
        report_to="comet_ml",
        seed=42,
    )

    # TODO: Handle this error: "Token indices sequence length is longer than the specified maximum sequence length 
    # for this model (2302 > 2048). Running this sequence through the model will result in indexing errors"
    model.config.use_cache = (
        False  # Gradient checkpointing is used by default but not compatible with caching
    )
    trainer = SFTTrainer(
    model=model,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Finish Comet ML experiment.
    experiment = comet_ml.get_global_experiment()
    experiment.end()

    return trainer, training_dataset, validation_dataset


def initialize():
    load_dotenv(find_dotenv())


    # Enable logging of model checkpoints
    os.environ["COMET_LOG_ASSETS"] = "True"


def compute_metrics(eval_pred: EvalPrediction):
    experiment = comet_ml.get_global_experiment()

    perplexity = np.exp(eval_pred.predictions.mean())

    if experiment:
        epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
        experiment.set_epoch(epoch)

    
    return {"perplexity": perplexity}
