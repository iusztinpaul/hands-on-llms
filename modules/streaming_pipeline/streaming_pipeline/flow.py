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
    """
    Builds a dataflow pipeline for processing news articles.

    Args:
        is_batch (bool): Whether the pipeline is processing a batch of articles or a stream.
        from_datetime (Optional[datetime.datetime]): The start datetime for processing articles.
        to_datetime (Optional[datetime.datetime]): The end datetime for processing articles.
        model_cache_dir (Optional[Path]): The directory to cache the embedding model.
        debug (bool): Whether to enable debug mode.

    Returns:
        Dataflow: The dataflow pipeline for processing news articles.
    """

    model = EmbeddingModelSingleton(cache_dir=model_cache_dir)
    is_input_mocked = debug is True and is_batch is False

    flow = Dataflow()
    flow.input(
        "input",
        _build_input(
            is_batch, from_datetime, to_datetime, is_input_mocked=is_input_mocked
        ),
    )
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
    is_input_mocked: bool = False,
) -> Input:
    if is_input_mocked is True:
        return TestingInput(mocked.financial_news)

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
