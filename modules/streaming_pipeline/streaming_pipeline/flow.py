from pathlib import Path
from typing import List, Optional

from bytewax.dataflow import Dataflow
from pydantic import parse_obj_as
from qdrant_client import QdrantClient

from streaming_pipeline.alpaca import AlpacaNewsInput
from streaming_pipeline.embeddings import EmbeddingModelSingleton
from streaming_pipeline.models import NewsArticle
from streaming_pipeline.qdrant import QdrantVectorOutput


def build(
    in_memory: bool = False,
    model_cache_dir: Optional[Path] = None,
) -> Dataflow:
    model = EmbeddingModelSingleton(cache_dir=model_cache_dir)

    flow = Dataflow()
    flow.input("input", AlpacaNewsInput(tickers=["*"]))
    flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
    flow.inspect(print)
    flow.map(lambda article: article.to_document())
    flow.map(lambda document: document.compute_chunks(model))
    flow.map(lambda document: document.compute_embeddings(model))

    if in_memory:
        flow.output(
            "output",
            QdrantVectorOutput(
                vector_size=model.max_input_length,
                client=QdrantClient(":memory:"),
            ),
        )
    else:
        flow.output(
            "output",
            QdrantVectorOutput(
                vector_size=model.max_input_length,
            ),
        )

    return flow
