import logging
import logging.config
import os
from pathlib import Path

import nvidia.cudnn
import yaml
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


def initialize(logging_config_path: str = "logging.yaml", env_file_path: str = ".env"):
    """
    Initializes the logger and environment variables.

    Args:
        logging_config_path (str): The path to the logging configuration file. Defaults to "logging.yaml".
        env_file_path (str): The path to the environment variables file. Defaults to ".env".
    """

    logger.info("Initializing logger...")
    try:
        initialize_logger(config_path=logging_config_path)
    except FileNotFoundError:
        logger.warning(
            f"No logging configuration file found at: {logging_config_path}. Setting logging level to INFO."
        )
        logging.basicConfig(level=logging.INFO)

    logger.info("Initializing env vars...")
    if env_file_path is None:
        env_file_path = find_dotenv(raise_error_if_not_found=True, usecwd=False)

    logger.info(f"Loading environment variables from: {env_file_path}")
    found_env_file = load_dotenv(env_file_path, verbose=True, override=True)
    if found_env_file is False:
        raise RuntimeError(f"Could not find environment file at: {env_file_path}")

    # Enable logging of model checkpoints.
    os.environ["COMET_LOG_ASSETS"] = "True"
    # Set to OFFLINE to run an Offline Experiment or DISABLE to turn off logging
    os.environ["COMET_MODE"] = "ONLINE"
    # Find out more about Comet ML configuration here: https://www.comet.com/docs/v2/integrations/ml-frameworks/huggingface/#configure-comet-for-hugging-face


def initialize_logger(
    config_path: str = "logging.yaml", logs_dir_name: str = "logs"
) -> logging.Logger:
    """Initialize logger from a YAML config file."""

    # Create logs directory.
    config_path_parent = Path(config_path).parent
    logs_dir = config_path_parent / logs_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "rt") as f:
        config = yaml.safe_load(f.read())

    # Make sure that existing logger will still work.
    config["disable_existing_loggers"] = False

    logging.config.dictConfig(config)


def initialize_cuda():
    print("##################### INITIALIZING CUDA #####################")
    """Initialize CUDA if available."""

    # Get the grandparent directory of the cudnn file path
    cudnn_file_path = nvidia.cudnn.__file__
    d_path = Path(cudnn_file_path).parent.parent

    # Start with the current LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    # Loop through each item in directory D
    for item in d_path.iterdir():
        if item.is_dir():
            lib_path = item / "lib"
            if lib_path.exists():
                # Append the lib subdirectory of the item to LD_LIBRARY_PATH
                ld_library_path += f":{lib_path}"

    # Update the LD_LIBRARY_PATH environment variable
    os.environ["LD_LIBRARY_PATH"] = ld_library_path

    # Optional: Print the updated LD_LIBRARY_PATH
    logger.info("Updated LD_LIBRARY_PATH:", ld_library_path)


initialize_cuda()
