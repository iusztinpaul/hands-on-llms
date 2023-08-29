
import json

from bytewax.dataflow import Dataflow
from qdrant_client import QdrantClient

from streaming_pipeline import initialize
from streaming_pipeline.alpaca import AlpacaNewsInput
from streaming_pipeline.documents import chunk, parse_article
from streaming_pipeline.embeddings import embedding
from streaming_pipeline.vector_db import QdrantVectorOutput

def build():
    flow = Dataflow()
    flow.input("input", AlpacaNewsInput(tickers=["*"]))
    flow.inspect(print)
    flow.flat_map(lambda x: json.loads(x))
    flow.map(parse_article)
    flow.map(chunk)
    flow.map(embedding)
    flow.inspect(print)
    flow.output("output", QdrantVectorOutput("test_collection", 384, client=QdrantClient(":memory:")))
    # flow.output(
    #     "output",
    #     QdrantVectorOutput(
    #         "test_collection",
    #         384,
    #         url=QDRANT_URL,
    #         api_key=QDRANT_API_KEY,
    #     ),
    # )
    
    return flow
