import json
import logging
import os
from typing import Optional

from bytewax.dataflow import Dataflow
from bytewax.inputs import DynamicInput, StatelessSource
from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from websocket import create_connection

from streaming_pipeline import initialize
from streaming_pipeline.documents import chunk, parse_article
from streaming_pipeline.embeddings import embedding

initialize()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Creating an object
logger = logging.getLogger()


class AlpacaSource(StatelessSource):
    def __init__(self, worker_tickers):
        # set the workers tickers
        self.worker_tickers = worker_tickers

        # establish a websocket connection to alpaca
        self.ws = create_connection("wss://stream.data.alpaca.markets/v1beta1/news")
        logger.info(self.ws.recv())

        # authenticate to the websocket
        self.ws.send(
            json.dumps(
                {"action": "auth", "key": f"{ALPACA_API_KEY}", "secret": f"{ALPACA_API_SECRET}"}
            )
        )
        logger.info(self.ws.recv())

        # subscribe to the tickers
        self.ws.send(json.dumps({"action": "subscribe", "news": self.worker_tickers}))
        logger.info(self.ws.recv())

    def next(self):
        return self.ws.recv()


class AlpacaNewsInput(DynamicInput):
    """Input class to receive streaming news data
    from the Alpaca real-time news API.

    Args:
        tickers: list - should be a list of tickers, use "*" for all
    """

    def __init__(self, tickers):
        self.TICKERS = tickers

    # distribute the tickers to the workers. If parallelized
    # workers will establish their own websocket connection and
    # subscribe to the tickers they are allocated
    def build(self, worker_index, worker_count):
        prods_per_worker = int(len(self.TICKERS) / worker_count)
        worker_tickers = self.TICKERS[
            int(worker_index * prods_per_worker) : int(
                worker_index * prods_per_worker + prods_per_worker
            )
        ]
        return AlpacaSource(worker_tickers)


def build_payloads(doc):
    payloads = []
    for c in doc.chunks:
        payload = doc.metadata
        payload.update({"text": c})
        payloads.append(payload)
    return payloads


class _QdrantVectorSink(StatelessSink):
    def __init__(self, client: QdrantClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def write(self, doc):
        _payloads = build_payloads(doc)
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(id=idx, vector=vector, payload=_payload)
                for idx, (vector, _payload) in enumerate(zip(doc.embeddings, _payloads))
            ],
        )


class QdrantVectorOutput(DynamicOutput):
    """Qdrant.

    Workers are the unit of parallelism.

    Can support at-least-once processing. Messages from the resume
    epoch will be duplicated right after resume.

    """

    def __init__(
        self,
        collection_name,
        vector_size,
        schema="",
        url="http://localhost:6333",
        api_key: Optional[str] = None,
        client=None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.schema = schema

        if client:
            self.client = client

        else:
            self.client = QdrantClient(url, api_key=api_key)

        try:
            self.client.get_collection(collection_name="test_collection")
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name="test_collection",
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
                schema=self.schema,
            )

    def build(self, worker_index, worker_count):
        return _QdrantVectorSink(self.client, self.collection_name)


flow = Dataflow()
flow.input("input", AlpacaNewsInput(tickers=["*"]))
flow.inspect(print)
flow.flat_map(lambda x: json.loads(x))
flow.map(parse_article)
flow.map(chunk)
flow.map(embedding)
flow.inspect(print)
# flow.output("output", QdrantVectorOutput("test_collection", 384, client=QdrantClient(":memory:")))
flow.output(
    "output",
    QdrantVectorOutput(
        "test_collection",
        384,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    ),
)
