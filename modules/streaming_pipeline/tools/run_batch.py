import datetime
import logging

from streaming_pipeline import initialize
from streaming_pipeline.flow import build as flow_builder


def build_flow(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
    latest_n_days: int = 4,
):
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    logger = logging.getLogger(__name__)

    to_datetime = datetime.datetime.now()
    from_datetime = to_datetime - datetime.timedelta(days=latest_n_days)
    logger.info(
        f"Extracting news from {from_datetime} to {to_datetime} [n_days={latest_n_days}]"
    )

    flow = flow_builder(
        in_memory=False,
        model_cache_dir=model_cache_dir,
        is_batch=True,
        from_datetime=from_datetime,
        to_datetime=to_datetime,
    )

    return flow
