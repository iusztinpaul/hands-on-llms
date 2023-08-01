import logging
import logging.config
import os
import yaml

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from training import api, constants, metrics, models


__all__ = ["api", "constants", "metrics", "models"]


logger = logging.getLogger(__name__)



def run_immediately_decorator(func):
    func()

    return func


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

    logging.config.dictConfig(config)


@run_immediately_decorator
def initialize(logging_config_path: str ="logging.yaml"):
     # Initialize logger.
    try:
        initialize_logger(config_path=logging_config_path)
    except FileNotFoundError:
        logger.warning(f"No logging configuration file found at: {logging_config_path}. Setting logging level to INFO.")
        logging.basicConfig(level=logging.INFO)

    logger.info("Initializing resources...")

    # Initialize environment variables.
    load_dotenv(find_dotenv())

    # Enable logging of model checkpoints
    os.environ["COMET_LOG_ASSETS"] = "True"
