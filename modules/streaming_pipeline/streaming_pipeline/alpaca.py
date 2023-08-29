import json
import logging
import os
from typing import List, Optional, Union

from bytewax.inputs import DynamicInput, StatelessSource
from bytewax.outputs import DynamicOutput, StatelessSink
from pydantic import parse_obj_as
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from websocket import create_connection

from streaming_pipeline import initialize
from streaming_pipeline.models import News

initialize()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Creating an object
logger = logging.getLogger()


class AlpacaNewsStream:
    ALPACA_NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

    # Alpaca Docs: https://alpaca.markets/docs/api-references/market-data-api/news-data/realtime/
    # Source of implementation inspiration: https://github.com/alpacahq/alpaca-py/blob/master/alpaca/common/websocket.py

    def __init__(self, tickers: Optional[List[str]] = None):
        if tickers is None:
            tickers = ["*"]

        self._tickers = tickers
        self._ws = None

    def start(self):
        self._connect()
        self._auth()

    def _connect(self):
        self._ws = create_connection(self.ALPACA_NEWS_STREAM_URL)

        msg = self.recv(serialize=False)

        if msg[0]["T"] != "success" or msg[0]["msg"] != "connected":
            raise ValueError("connected message not received")
        else:
            logger.info("[AlpacaNewsStream]: Connected to Alpaca News Stream.")

    def _auth(self):
        self._ws.send(
            self._build_message(
                {
                    "action": "auth",
                    "key": f"{ALPACA_API_KEY}",
                    "secret": f"{ALPACA_API_SECRET}",
                }
            )
        )

        msg = self.recv(serialize=False)
        if msg[0]["T"] == "error":
            raise ValueError(msg[0].get("msg", "auth failed"))
        elif msg[0]["T"] != "success" or msg[0]["msg"] != "authenticated":
            raise ValueError("failed to authenticate")
        else:
            logger.info("[AlpacaNewsStream]: Authenticated with Alpaca News Stream.")

    def subscribe(self):
        self._ws.send(
            self._build_message({"action": "subscribe", "news": self._tickers})
        )

        msg = self.recv(serialize=False)
        if msg[0]["T"] != "subscription":
            raise ValueError("failed to subscribe")
        else:
            logger.info("[AlpacaNewsStream]: Subscribed to Alpaca News Stream.")

    def ubsubscribe(self):
        self._ws.send(
            self._build_message({"action": "unsubscribe", "news": self._tickers})
        )

        msg = self.recv(serialize=False)
        if msg[0]["T"] != "subscription":
            raise ValueError("failed to unsubscribe")
        else:
            logger.info("[AlpacaNewsStream]: Unsubscribed from Alpaca News Stream.")

    def _build_message(self, message: dict) -> str:
        return json.dumps(message)

    def recv(self, serialize: bool = True) -> Union[dict, List[News]]:
        if self._ws:
            message = self._ws.recv()
            logger.info(f"[AlpacaNewsStream]: Received message: {message}")
            message = json.loads(message)

            if serialize is True:
                message = parse_obj_as(List[News], message)

            return message
        else:
            raise RuntimeError("Websocket not initialized. Call start() first.")

    def close(self) -> None:
        if self._ws:
            self._ws.close()
            self._ws = None


class AlpacaSource(StatelessSource):
    def __init__(self, worker_tickers):
        self._alpaca_client = AlpacaNewsStream(tickers=worker_tickers)
        self._alpaca_client.start()
        self._alpaca_client.subscribe()

    def next(self):
        return self._alpaca_client.recv()

    def close(self):
        self._alpaca_client.ubsubscribe()

        return self._alpaca_client.close()


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


import time

alpaca_client = AlpacaNewsStream(tickers=["*"])
alpaca_client.start()
alpaca_client.subscribe()
while True:
    alpaca_client.recv()
    time.sleep(1)
