import logging
import os
from typing import List, Optional

from bytewax.inputs import DynamicInput, StatelessSource

# Creating an object
logger = logging.getLogger()


class AlpacaNewsBatchInput(DynamicInput):
    """Input class to receive batch news data
    from the Alpaca news RESTful API.

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

        return AlpacaNewsBatchSource(tickers=worker_tickers)


class AlpacaNewsBatchSource(StatelessSource):
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
) -> "AlpacaNewsBatchClient":
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

    return AlpacaNewsBatchClient(
        api_key=api_key, api_secret=api_secret, tickers=tickers
    )


class AlpacaNewsBatchClient:
    def __init__(self, api_key: str, api_secret: str, tickers: List[str]):
        self._api_key = api_key
        self._api_secret = api_secret
        self._tickers = tickers
