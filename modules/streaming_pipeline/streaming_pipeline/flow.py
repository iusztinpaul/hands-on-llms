import datetime
from pathlib import Path
from typing import List, Optional

from bytewax.dataflow import Dataflow
from bytewax.inputs import Input
from bytewax.outputs import Output
from bytewax.testing import TestingInput
from pydantic import parse_obj_as
from qdrant_client import QdrantClient
from streaming_pipeline import mocked

from streaming_pipeline.alpaca_batch import AlpacaNewsBatchInput
from streaming_pipeline.alpaca_stream import AlpacaNewsStreamInput
from streaming_pipeline.embeddings import EmbeddingModelSingleton
from streaming_pipeline.models import NewsArticle
from streaming_pipeline.qdrant import QdrantVectorOutput


def build(
    is_batch: bool = False,
    from_datetime: Optional[datetime.datetime] = None,
    to_datetime: Optional[datetime.datetime] = None,
    model_cache_dir: Optional[Path] = None,
    debug: bool = False,
) -> Dataflow:
    model = EmbeddingModelSingleton(cache_dir=model_cache_dir)

    flow = Dataflow()
    flow.input("input", _build_input(is_batch, from_datetime, to_datetime, dummy_input=debug))
    flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
    if debug:
        flow.inspect(print)
    flow.map(lambda article: article.to_document())
    flow.map(lambda document: document.compute_chunks(model))
    flow.map(lambda document: document.compute_embeddings(model))
    flow.output("output", _build_output(model, in_memory=debug))

    return flow


def _build_input(
    is_batch: bool = False,
    from_datetime: Optional[datetime.datetime] = None,
    to_datetime: Optional[datetime.datetime] = None,
    dummy_input: bool = False
) -> Input:
    if dummy_input is True and is_batch is False:
        return TestingInput(mocked.news)
        
    if is_batch:
        assert (
            from_datetime is not None and to_datetime is not None
        ), "from_datetime and to_datetime must be provided when is_batch is True"

        return AlpacaNewsBatchInput(
            from_datetime=from_datetime, to_datetime=to_datetime, tickers=["*"]
        )
    else:
        return AlpacaNewsStreamInput(tickers=["*"])


def _build_output(model: EmbeddingModelSingleton, in_memory: bool = False) -> Output:
    if in_memory:
        return QdrantVectorOutput(
            vector_size=model.max_input_length,
            client=QdrantClient(":memory:"),
        )
    else:
        return QdrantVectorOutput(
            vector_size=model.max_input_length,
        )
