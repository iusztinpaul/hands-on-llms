import logging
import logging.config
from pathlib import Path

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
        env_file_path = find_dotenv(raise_error_if_not_found=False, usecwd=False)

    if env_file_path is not None:
        logger.info(f"Loading environment variables from: {env_file_path}")
        load_dotenv(env_file_path, verbose=True, override=True)


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
