import json
import logging
import os
from typing import List, Optional, Union

from bytewax.inputs import DynamicInput, StatelessSource
from websocket import create_connection

# Creating an object
logger = logging.getLogger()


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

        return AlpacaSource(tickers=worker_tickers)


class AlpacaSource(StatelessSource):
    def __init__(self, tickers: List[str]):
        self._alpaca_client = build_alpaca_client(tickers=tickers)
        self._alpaca_client.start()
        self._alpaca_client.subscribe()

    def next(self):
        return self._alpaca_client.recv()

    def close(self):
        self._alpaca_client.ubsubscribe()

        return self._alpaca_client.close()
    
    
def build_alpaca_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> 'AlpacaNewsStreamClient':
    if api_key is None:
        try:
            api_key = os.environ["ALPACA_API_KEY"]
        except KeyError:
            raise KeyError(
                "ALPACA_API_KEY must be set as environment variable or manually passed as an argument."
            )

    if api_secret is None:
        try:
            api_secret = os.environ["ALPACA_API_SECRET"]
        except KeyError:
            raise KeyError(
                "ALPACA_API_SECRET must be set as environment variable or manually passed as an argument."
            )

    if tickers is None:
        tickers = ["*"]

    return AlpacaNewsStreamClient(
        api_key=api_key, api_secret=api_secret, tickers=tickers
    )
    
    
class AlpacaNewsStreamClient:
    NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

    # Alpaca Docs: https://alpaca.markets/docs/api-references/market-data-api/news-data/realtime/
    # Source of implementation inspiration: https://github.com/alpacahq/alpaca-py/blob/master/alpaca/common/websocket.py

    def __init__(self, api_key: str, api_secret: str, tickers: List[str]):
        self._api_key = api_key
        self._api_secret = api_secret
        self._tickers = tickers
        self._ws = None

    def start(self):
        self._connect()
        self._auth()

    def _connect(self):
        self._ws = create_connection(self.NEWS_STREAM_URL)

        msg = self.recv()

        if msg[0]["T"] != "success" or msg[0]["msg"] != "connected":
            raise ValueError("connected message not received")
        else:
            logger.info("[AlpacaNewsStream]: Connected to Alpaca News Stream.")

    def _auth(self):
        self._ws.send(
            self._build_message(
                {
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._api_secret,
                }
            )
        )

        msg = self.recv()
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

        msg = self.recv()
        if msg[0]["T"] != "subscription":
            raise ValueError("failed to subscribe")
        else:
            logger.info("[AlpacaNewsStream]: Subscribed to Alpaca News Stream.")

    def ubsubscribe(self):
        self._ws.send(
            self._build_message({"action": "unsubscribe", "news": self._tickers})
        )

        msg = self.recv()
        if msg[0]["T"] != "subscription":
            raise ValueError("failed to unsubscribe")
        else:
            logger.info("[AlpacaNewsStream]: Unsubscribed from Alpaca News Stream.")

    def _build_message(self, message: dict) -> str:
        return json.dumps(message)

    def recv(self) -> Union[dict, List[dict]]:
        if self._ws:
            message = self._ws.recv()
            logger.info(f"[AlpacaNewsStream]: Received message: {message}")
            message = json.loads(message)

            return message
        else:
            raise RuntimeError("Websocket not initialized. Call start() first.")

    def close(self) -> None:
        if self._ws:
            self._ws.close()
            self._ws = None
