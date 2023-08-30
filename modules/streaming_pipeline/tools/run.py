from bytewax.dataflow import Dataflow

from streaming_pipeline import initialize
from streaming_pipeline.flow import build as flow_builder


def run() -> Dataflow:
    initialize()

    flow = flow_builder()

    return flow
