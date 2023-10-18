import logging

import fire
from beam import App, Image, Runtime, Volume, VolumeType

from financial_bot import load_bot

logger = logging.getLogger(__name__)


# === Beam Apps ===

financial_bot = App(
    name="financial_bot",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="T4",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)
financial_bot_dev = App(
    name="financial_bot_dev",
    runtime=Runtime(
        cpu=4,
        memory="4Gi",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)


# === Bot Loaders ===


def load_bot_dev(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
):
    """Load the Financial Assistant Bot in dev mode: the embedding model runs on CPU and the LLM is mocked."""

    return load_bot(
        env_file_path=env_file_path,
        logging_config_path=logging_config_path,
        model_cache_dir=model_cache_dir,
        embedding_model_device="cpu",
        debug=True,
    )


# === Bot Runners ===

# TODO: Test if the Beam decorator can use the "load_bot" function imported from the module.
@financial_bot.rest_api(keep_warm_seconds=300, loader=load_bot)
def run(**inputs):
    """Run the bot under the Beam RESTful API endpoint."""

    response = _run(**inputs)

    return response


@financial_bot_dev.rest_api(keep_warm_seconds=300, loader=load_bot_dev)
def run_dev(**inputs):
    """Run the bot under the Beam RESTful API endpoint [Dev Mode]."""

    response = _run(**inputs)

    return response


def run_local(about_me: str, question: str, debug: bool = False):
    """Run the bot locally in production or dev mode."""

    if debug is True:
        bot = load_bot_dev(model_cache_dir=None)
    else:
        bot = load_bot(model_cache_dir=None)

    inputs = {"about_me": about_me, "question": question, "context": bot}

    response = _run(**inputs)

    return response


def _run(**inputs):
    """Central function that calls the bot and returns the response"""

    from financial_bot import utils

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    bot = inputs["context"]
    input_payload = {
        "about_me": inputs["about_me"],
        "question": inputs["question"],
    }
    response = bot.answer(**input_payload)

    return response


if __name__ == "__main__":
    fire.Fire(run_local)
