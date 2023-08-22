import logging
import os
from pathlib import Path
import time
from typing import Optional, Tuple

import comet_llm

from datasets import Dataset
from tqdm import tqdm
from training_pipeline.data import finqa, utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig

from training_pipeline.configs import InferenceConfig
from training_pipeline import constants, models


try:
    comet_project_name = os.environ["COMET_PROJECT_NAME"]
except KeyError:
    raise RuntimeError("Please set the COMET_PROJECT_NAME environment variable.")


logger = logging.getLogger(__name__)


class FinQAInferenceAPI:
    def __init__(
        self,
        peft_model_id: str,
        model_id: str,
        name: str = "inference-prompts",
        root_dataset_dir: Optional[Path] = None,
        max_new_tokens: int = 50,
        model_cache_dir: Optional[Path] = None,
        debug: bool = False,
        device: str = "cuda:0",
    ):
        self._peft_model_id = peft_model_id
        self._model_id = model_id
        self._name = name
        self._root_dataset_dir = root_dataset_dir
        self._max_new_tokens = max_new_tokens
        self._model_cache_dir = model_cache_dir
        self._debug = debug
        self._device = device

        self._model, self._tokenizer, self._peft_config = self.load_model()
        if self._root_dataset_dir is not None:
            self._dataset = self.load_data()
        else:
            self._dataset = None

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        root_dataset_dir: Optional[Path] = None,
        model_cache_dir: Optional[Path] = None,
    ):
        return cls(
            peft_model_id=config.peft_model["id"],
            model_id=config.model["id"],
            root_dataset_dir=root_dataset_dir,
            max_new_tokens=config.model["max_new_tokens"],
            model_cache_dir=model_cache_dir,
            debug=config.setup.get("debug", False),
            device=config.setup.get("device", "cuda:0"),
        )

    def load_data(self) -> Dataset:
        logger.info(f"Loading FinQA datasets from {self._root_dataset_dir=}")

        if self._debug:
            max_samples = 3
        else:
            max_samples = 20

        dataset = finqa.FinQADataset(
            data_path=self._root_dataset_dir / "private_test.json",
            scope=constants.Scope.INFERENCE,
            max_samples=max_samples,
        ).to_huggingface()

        logger.info(f"Loaded {len(dataset)} samples for inference")

        return dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        logger.info(f"Loading model using {self._model_id=} and {self._peft_model_id=}")

        model, tokenizer, peft_config = models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            peft_pretrained_model_name_or_path=self._peft_model_id,
            gradient_checkpointing=False,
            cache_dir=self._model_cache_dir,
        )

        return model, tokenizer, peft_config

    def infer(self, question: str) -> str:
        # TODO: Handle this error: "Token indices sequence length is longer than the specified maximum sequence length
        # for this model (2302 > 2048). Running this sequence through the model will result in indexing errors"

        start_time = time.time()
        answer = models.prompt(
            model=self._model,
            tokenizer=self._tokenizer,
            input_text=question,
            max_new_tokens=self._max_new_tokens,
            device=self._device,
            return_only_answer=True,
        )
        end_time = time.time()

        duration_milliseconds = (end_time - start_time) * 1000

        comet_llm.log_prompt(
            project=f"{comet_project_name}-{self._name}",
            prompt=question,
            output=answer,
            metadata={
                "usage.prompt_tokens": len(question),
                "usage.total_tokens": len(question) + len(answer),
                "usage.max_new_tokens": self._max_new_tokens,
                "usage.actual_new_tokens": len(answer),
                "model": self._model_id,
                "peft_model": self._peft_model_id,
            },
            duration=duration_milliseconds,
        )

        return answer

    def infer_all(self, output_file: Optional[Path] = None) -> None:
        assert (
            self._dataset is not None
        ), "Dataset not loaded. Provide a dataset directory to the constructor: 'root_dataset_dir'."

        question_and_answers = []
        should_save_output = output_file is not None
        for sample in tqdm(self._dataset):
            answer = self.infer(question=sample["text"])

            if should_save_output:
                question_and_answers.append(
                    {
                        "question": sample["text"],
                        "answer": answer,
                    }
                )

        if should_save_output:
            utils.write_json(question_and_answers, output_file)
