import logging
import logging.config
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


def load_bot(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
    embedding_model_device: str = "cuda:0",
    debug: bool = False,
):
    """
    Load the financial assistant bot in production or development mode based on the `debug` flag
    
    production: the embedding model runs on GPU and the fine-tuned LLM is used.
    dev: the embedding model runs on CPU and the fine-tuned LLM is mocked.
    """

    from financial_bot import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from financial_bot import utils
    from financial_bot.langchain_bot import FinancialBot

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    bot = FinancialBot(
        model_cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        embedding_model_device=embedding_model_device,
        debug=debug,
    )

    return bot


def initialize(logging_config_path: str = "logging.yaml", env_file_path: str = ".env"):
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
