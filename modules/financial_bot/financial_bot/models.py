import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from comet_ml import API
from langchain.llms import HuggingFacePipeline
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)

from financial_bot import constants
from financial_bot.utils import MockedPipeline

logger = logging.getLogger(__name__)


def download_from_model_registry(model_id: str, cache_dir: Optional[Path] = None):
    if cache_dir is None:
        cache_dir = constants.CACHE_DIR
    output_folder = cache_dir / "models" / model_id

    workspace, model_id = model_id.split("/")
    model_name, version = model_id.split(":")

    api = API()
    model = api.get_model(workspace=workspace, model_name=model_name)
    model.download(version=version, output_folder=output_folder, expand=True)

    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"There should be only one directory inside the model folder. \
                Check the downloaded model at: {output_folder}"
        )

    logger.info(f"Model {model_id=} downloaded from the registry to: {model_dir}")

    return model_dir


def build_huggingface_pipeline(
    llm_model_id: str,
    llm_lora_model_id: str,
    max_new_tokens: int = constants.LLM_INFERNECE_MAX_NEW_TOKENS,
    temperature: float = constants.LLM_INFERENCE_TEMPERATURE,
    gradient_checkpointing: bool = False,
    use_streamer: bool = False,
    cache_dir: Optional[Path] = None,
    debug: bool = False,
):
    """Using our custom LLM + Finetuned checkpoint we create a HF pipeline"""

    if debug is True:
        return HuggingFacePipeline(
            pipeline=MockedPipeline(f=lambda _: "You are doing great!")
        )

    model, tokenizer, _ = build_qlora_model(
        pretrained_model_name_or_path=llm_model_id,
        peft_pretrained_model_name_or_path=llm_lora_model_id,
        gradient_checkpointing=gradient_checkpointing,
        cache_dir=cache_dir,
    )
    model.eval()

    if use_streamer:
        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
    else:
        streamer = None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        streamer=streamer,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    if use_streamer:
        return hf, streamer

    return hf


def build_qlora_model(
    pretrained_model_name_or_path: str = "tiiuae/falcon-7b-instruct",
    peft_pretrained_model_name_or_path: Optional[str] = None,
    gradient_checkpointing: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
    """
    Function that builds a QLoRA LLM model based on the given HuggingFace name:
        1.   Create and prepare the bitsandbytes configuration for QLoRa's quantization
        2.   Download, load, and quantize on-the-fly Falcon-7b
        3.   Create and prepare the LoRa configuration
        4.   Load and configuration Falcon-7B's tokenizer
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        quantization_config=bnb_config,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # TODO: Should we also enable kbit training? Check out what it does.
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=False,
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if peft_pretrained_model_name_or_path:
        is_model_name = not os.path.isdir(peft_pretrained_model_name_or_path)
        if is_model_name:
            logger.info(
                f"Downloading {peft_pretrained_model_name_or_path} from CometML Model Registry:"
            )
            peft_pretrained_model_name_or_path = download_from_model_registry(
                model_id=peft_pretrained_model_name_or_path,
                cache_dir=cache_dir,
            )

        logger.info(f"Loading Lora Confing from: {peft_pretrained_model_name_or_path}")
        lora_config = LoraConfig.from_pretrained(peft_pretrained_model_name_or_path)
        assert (
            lora_config.base_model_name_or_path == pretrained_model_name_or_path
        ), f"Lora Model trained on different base model than the one requested: \
        {lora_config.base_model_name_or_path} != {pretrained_model_name_or_path}"

        logger.info(f"Loading Peft Model from: {peft_pretrained_model_name_or_path}")
        model = PeftModel.from_pretrained(model, peft_pretrained_model_name_or_path)
    else:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["query_key_value"],
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # Gradient checkpointing is not compatible with caching.
        )
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True  # It is good practice to enable caching when using the model for inference.

    return model, tokenizer, lora_config
