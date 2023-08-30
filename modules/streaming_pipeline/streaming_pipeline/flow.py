from typing import List

from bytewax.dataflow import Dataflow
from pydantic import parse_obj_as
from qdrant_client import QdrantClient

from streaming_pipeline.alpaca import AlpacaNewsInput
from streaming_pipeline.documents import chunk, parse_article
from streaming_pipeline.embeddings import compute_embeddings
from streaming_pipeline.models import NewsArticle
from streaming_pipeline.vector_db import QdrantVectorOutput


def build(in_memory: bool = False) -> Dataflow:
    flow = Dataflow()
    flow.input("input", AlpacaNewsInput(tickers=["*"]))
    flow.inspect(print)
    flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
    flow.map(parse_article)
    flow.map(chunk)
    flow.map(compute_embeddings)
    flow.inspect(print)

    if in_memory:
        flow.output(
            "output",
            QdrantVectorOutput("test_collection", 384, client=QdrantClient(":memory:")),
        )
    else:
        flow.output(
            "output",
            QdrantVectorOutput(
                "test_collection",
                384,
            ),
        )

    return flow
