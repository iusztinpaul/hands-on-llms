import logging
from pathlib import Path

import fire
from beam import App, Image, Runtime, Volume, VolumeType

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


logger = logging.getLogger(__name__)


def load_models(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
):
    from financial_bot import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)
    
    from financial_bot import utils
    from financial_bot.langchain_bot import FinancialBot
    
    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)
    
    bot = FinancialBot(model_cache_dir=Path(model_cache_dir))
    
    return bot


@financial_bot.rest_api(keep_warm_seconds=300, loader=load_models)
def run(**inputs):
    from financial_bot import utils

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)
    
    # TODO: Check how memory is persisted between requests.
    bot = inputs["context"]
    input_payload = {
        "about_me": inputs["about_me"],
        "question": inputs["question"],
    }
    response = bot.answer(**input_payload)

    return response


@financial_bot.rest_api(keep_warm_seconds=300, loader=load_models)
def run_dev(**inputs):
    from financial_bot import utils

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)
    
    bot = inputs["context"]
    
    input_payload = {
        "about_me": "I'm a student and I have some money that I want to invest.",
        "question": "Should I consider investing in stocks from the Tech Sector?",
    }
    response = bot.answer(**input_payload)
    print(response)

    next_question = "What about the Energy Sector?"
    input_payload["question"] = next_question
    response = bot.answer(**input_payload)
    print(response)

    return response


if __name__ == "__main__":
    fire.Fire(run)
