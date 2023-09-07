import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from comet_ml import API
from peft import LoraConfig, PeftConfig, PeftModel
from training_pipeline import constants
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
