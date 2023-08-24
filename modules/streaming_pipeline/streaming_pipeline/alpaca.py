import json
import logging
import os

from bytewax.dataflow import Dataflow
from bytewax.inputs import DynamicInput, StatelessSource
from websocket import create_connection

from streaming_pipeline import initialize
from streaming_pipeline.documents import chunk, parse_article
from streaming_pipeline.embeddings import embedding

initialize()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

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
                {"action": "auth", "key": f"{API_KEY}", "secret": f"{API_SECRET}"}
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


flow = Dataflow()
flow.input("input", AlpacaNewsInput(tickers=["*"]))
flow.inspect(print)
flow.flat_map(lambda x: json.loads(x))
flow.map(parse_article)
flow.map(chunk)
flow.map(embedding)
flow.inspect(print)
