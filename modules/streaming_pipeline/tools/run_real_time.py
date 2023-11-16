from streaming_pipeline import initialize
from streaming_pipeline.flow import build as flow_builder


def build_flow(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
    debug: bool = False,
):
    """
    Builds a Bytewax flow for real-time data processing.

    Args:
        env_file_path (str, optional): Path to the environment file. Defaults to ".env".
        logging_config_path (str, optional): Path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str, optional): Path to the directory where the model cache is stored. Defaults to None.
        debug (bool, optional): Whether to run the flow in debug mode. Defaults to False.

    Returns:
        flow (prefect.Flow): The Bytewax flow for real-time data processing.
    """

    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    flow = flow_builder(model_cache_dir=model_cache_dir, debug=debug)

    return flow
