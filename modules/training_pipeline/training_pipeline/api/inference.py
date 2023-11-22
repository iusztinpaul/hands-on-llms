import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import comet_llm
from datasets import Dataset
from peft import PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_pipeline import constants, models
from training_pipeline.configs import InferenceConfig
from training_pipeline.data import qa, utils
from training_pipeline.prompt_templates.prompter import get_llm_template

try:
    comet_project_name = os.environ["COMET_PROJECT_NAME"]
except KeyError:
    raise RuntimeError("Please set the COMET_PROJECT_NAME environment variable.")

logger = logging.getLogger(__name__)


class InferenceAPI:
    """
    A class for performing inference using a trained LLM model.

    Args:
        peft_model_id (str): The ID of the PEFT model to use.
        model_id (str): The ID of the LLM model to use.
        template_name (str): The name of the LLM template to use.
        root_dataset_dir (Path): The root directory of the dataset to use.
        test_dataset_file (Path): The path to the test dataset file.
        name (str, optional): The name of the inference API. Defaults to "inference-api".
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 50.
        temperature (float, optional): The temperature to use when generating new tokens. Defaults to 1.0.
        model_cache_dir (Path, optional): The directory to use for caching the LLM model.
            Defaults to constants.CACHE_DIR.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        device (str, optional): The device to use for inference. Defaults to "cuda:0".

    """

    def __init__(
        self,
        peft_model_id: str,
        model_id: str,
        template_name: str,
        root_dataset_dir: Path,
        test_dataset_file: Path,
        name: str = "inference-api",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        model_cache_dir: Path = constants.CACHE_DIR,
        debug: bool = False,
        device: str = "cuda:0",
    ):
        self._template_name = template_name
        self._prompt_template = get_llm_template(template_name)
        self._peft_model_id = peft_model_id
        self._model_id = model_id
        self._name = name
        self._root_dataset_dir = root_dataset_dir
        self._test_dataset_file = test_dataset_file
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
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
        root_dataset_dir: Path,
        model_cache_dir: Path = constants.CACHE_DIR,
    ):
        """
        Creates an instance of the InferenceAPI class from an InferenceConfig object.

        Args:
            config (InferenceConfig): The InferenceConfig object to use.
            root_dataset_dir (Path): The root directory of the dataset to use.
            model_cache_dir (Path, optional): The directory to use for caching the LLM model.
                Defaults to constants.CACHE_DIR.

        Returns:
            InferenceAPI: An instance of the InferenceAPI class.

        """

        return cls(
            peft_model_id=config.peft_model["id"],
            model_id=config.model["id"],
            template_name=config.model["template_name"],
            root_dataset_dir=root_dataset_dir,
            test_dataset_file=config.dataset["file"],
            max_new_tokens=config.model["max_new_tokens"],
            temperature=config.model["temperature"],
            model_cache_dir=model_cache_dir,
            debug=config.setup.get("debug", False),
            device=config.setup.get("device", "cuda:0"),
        )

    def load_data(self) -> Dataset:
        """
        Loads the QA dataset.

        Returns:
            Dataset: The loaded QA dataset.

        """

        logger.info(f"Loading QA dataset from {self._root_dataset_dir=}")

        if self._debug:
            max_samples = 3
        else:
            max_samples = None

        dataset = qa.FinanceDataset(
            data_path=self._root_dataset_dir / self._test_dataset_file,
            template=self._template_name,
            scope=constants.Scope.INFERENCE,
            max_samples=max_samples,
        ).to_huggingface()

        logger.info(f"Loaded {len(dataset)} samples for inference")

        return dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        """
        Loads the LLM model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the loaded LLM model, tokenizer,
                and PEFT config.

        """

        logger.info(f"Loading model using {self._model_id=} and {self._peft_model_id=}")

        model, tokenizer, peft_config = models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            peft_pretrained_model_name_or_path=self._peft_model_id,
            gradient_checkpointing=False,
            cache_dir=self._model_cache_dir,
        )
        model.eval()

        return model, tokenizer, peft_config

    def infer(self, infer_prompt: str, infer_payload: dict) -> str:
        """
        Performs inference using the loaded LLM model.

        Args:
            infer_prompt (str): The prompt to use for inference.
            infer_payload (dict): The payload to use for inference.

        Returns:
            str: The generated answer.

        """

        start_time = time.time()
        answer = models.prompt(
            model=self._model,
            tokenizer=self._tokenizer,
            input_text=infer_prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            device=self._device,
            return_only_answer=True,
        )
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if not self._debug:
            comet_llm.log_prompt(
                project=f"{comet_project_name}-{self._name}-monitor-prompts",
                prompt=infer_prompt,
                output=answer,
                prompt_template=self._prompt_template.infer_raw_template,
                prompt_template_variables=infer_payload,
                # TODO: Count tokens instead of using len().
                metadata={
                    "usage.prompt_tokens": len(infer_prompt),
                    "usage.total_tokens": len(infer_prompt) + len(answer),
                    "usage.max_new_tokens": self._max_new_tokens,
                    "usage.actual_new_tokens": len(answer),
                    "model": self._model_id,
                    "peft_model": self._peft_model_id,
                },
                duration=duration_milliseconds,
            )

        return answer

    def infer_all(self, output_file: Optional[Path] = None) -> None:
        """
        Performs inference on all samples in the loaded dataset.

        Args:
            output_file (Optional[Path], optional): The file to save the output to. Defaults to None.

        """

        assert (
            self._dataset is not None
        ), "Dataset not loaded. Provide a dataset directory to the constructor: 'root_dataset_dir'."

        prompt_and_answers = []
        should_save_output = output_file is not None
        for sample in tqdm(self._dataset):
            answer = self.infer(
                infer_prompt=sample["prompt"], infer_payload=sample["payload"]
            )

            if should_save_output:
                prompt_and_answers.append(
                    {
                        "prompt": sample["prompt"],
                        "answer": answer,
                    }
                )

        if should_save_output:
            utils.write_json(prompt_and_answers, output_file)
