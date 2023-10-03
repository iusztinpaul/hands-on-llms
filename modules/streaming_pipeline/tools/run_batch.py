import datetime
from typing import Optional
from bytewax.testing import run_main

from streaming_pipeline import initialize
from streaming_pipeline.flow import build as flow_builder


def build_flow(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
    latest_n_days: int = 4,
):
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    to_datetime = datetime.datetime.now()
    flow = flow_builder(
        in_memory=True,
        model_cache_dir=model_cache_dir,
        is_batch=True,
        from_datetime=to_datetime - datetime.timedelta(days=latest_n_days),
        to_datetime=to_datetime,
        )

    return flow
