import datetime
import logging
import os
from typing import List, Optional

import requests
from bytewax.inputs import DynamicInput, StatelessSource

from streaming_pipeline import utils

logger = logging.getLogger()


class AlpacaNewsBatchInput(DynamicInput):
    """Input class to receive batch news data
    from the Alpaca news RESTful API.

    Args:
        tickers: list - should be a list of tickers, use "*" for all
        from_datetime: datetime.datetime - the start datetime for the news data
        to_datetime: datetime.datetime - the end datetime for the news data
    """

    def __init__(
        self,
        tickers: List[str],
        from_datetime: datetime.datetime,
        to_datetime: datetime.datetime,
    ):
        self._tickers = tickers
        self._from_datetime = from_datetime
        self._to_datetime = to_datetime

    def build(self, worker_index, worker_count):
        # Distribute different time ranges to different workers,
        # based on the total number of workers.
        datetime_intervals = utils.split_time_range_into_intervals(
            from_datetime=self._from_datetime,
            to_datetime=self._to_datetime,
            n=worker_count,
        )
        worker_datetime_interval = datetime_intervals[worker_index]
        worker_from_datetime, worker_to_datetime = worker_datetime_interval

        logger.info(
            f"woker_index: {worker_index} start from {worker_from_datetime} to {worker_to_datetime}"
        )

        return AlpacaNewsBatchSource(
            tickers=self._tickers,
            from_datetime=worker_from_datetime,
            to_datetime=worker_to_datetime,
        )


class AlpacaNewsBatchSource(StatelessSource):
    """
    A batch source for retrieving news articles from Alpaca.

    Args:
        tickers (List[str]): A list of ticker symbols to retrieve news for.
        from_datetime (datetime.datetime): The start datetime to retrieve news from.
        to_datetime (datetime.datetime): The end datetime to retrieve news from.
    """

    def __init__(
        self,
        tickers: List[str],
        from_datetime: datetime.datetime,
        to_datetime: datetime.datetime,
    ):
        self._alpaca_client = build_alpaca_client(
            from_datetime=from_datetime, to_datetime=to_datetime, tickers=tickers
        )

    def next(self):
        """
        Retrieves the next batch of news articles.

        Returns:
            List[dict]: A list of news articles.
        """

        news = self._alpaca_client.list()

        if news is None or len(news) == 0:
            raise StopIteration()

        return news

    def close(self):
        """
        Closes the batch source.
        """

        pass


def build_alpaca_client(
    from_datetime: datetime.datetime,
    to_datetime: datetime.datetime,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> "AlpacaNewsBatchClient":
    """
    Builds an AlpacaNewsBatchClient object with the specified parameters.

    Args:
        from_datetime (datetime.datetime): The start datetime for the news batch.
        to_datetime (datetime.datetime): The end datetime for the news batch.
        api_key (Optional[str], optional): The Alpaca API key. Defaults to None.
        api_secret (Optional[str], optional): The Alpaca API secret. Defaults to None.
        tickers (Optional[List[str]], optional): The list of tickers to retrieve news for. Defaults to None.

    Raises:
        KeyError: If api_key or api_secret is not provided and is not found in the environment variables.

    Returns:
        AlpacaNewsBatchClient: The AlpacaNewsBatchClient object.
    """

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
        from_datetime=from_datetime,
        to_datetime=to_datetime,
        api_key=api_key,
        api_secret=api_secret,
        tickers=tickers,
    )


class AlpacaNewsBatchClient:
    """
    Alpaca News API Client that uses a RESTful API to fetch news data.

    Attributes:
        NEWS_URL (str): The URL for the Alpaca News API.
        _from_datetime (datetime.datetime): The start datetime for the news data.
        _to_datetime (datetime.datetime): The end datetime for the news data.
        _api_key (str): The API key for the Alpaca News API.
        _api_secret (str): The API secret for the Alpaca News API.
        _tickers (List[str]): A list of tickers to filter the news data.
        _page_token (str): The page token for the next page of news data.
        _first_request (bool): A flag indicating whether this is the first request for news data.
    """

    NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(
        self,
        from_datetime: datetime.datetime,
        to_datetime: datetime.datetime,
        api_key: str,
        api_secret: str,
        tickers: List[str],
    ):
        """
        Initializes a new instance of the AlpacaNewsBatchClient class.

        Args:
            from_datetime (datetime.datetime): The start datetime for the news data.
            to_datetime (datetime.datetime): The end datetime for the news data.
            api_key (str): The API key for the Alpaca News API.
            api_secret (str): The API secret for the Alpaca News API.
            tickers (List[str]): A list of tickers to filter the news data.
        """

        self._from_datetime = from_datetime
        self._to_datetime = to_datetime
        self._api_key = api_key
        self._api_secret = api_secret
        self._tickers = tickers

        self._page_token = None
        self._first_request = True

    @property
    def try_request(self) -> bool:
        """
        A property indicating whether a request should be attempted.

        Returns:
            bool: True if a request should be attempted, False otherwise.
        """

        return self._first_request or self._page_token is not None

    def list(self):
        """
        Convenience function to fetch a batch of news from Alpaca API

        Returns:
            List[Dict]: A list of news items.
        """

        if not self.try_request:
            return None

        self._first_request = False

        # prepare the request URL
        headers = {
            "Apca-Api-Key-Id": self._api_key,
            "Apca-Api-Secret-Key": self._api_secret,
        }

        # Look at all the parameters here: https://alpaca.markets/docs/api-references/market-data-api/news-data/historical/
        # or here: https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/requests.py#L357
        params = {
            "start": self._from_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": self._to_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "include_content": True,
            "sort": "ASC",
        }
        if self._page_token is not None:
            params["page_token"] = self._page_token

        response = requests.get(self.NEWS_URL, headers=headers, params=params)

        # parse output
        next_page_token = None
        if response.status_code == 200:  # Check if the request was successful
            # parse response into json
            news_json = response.json()

            # extract next page token (if any)
            next_page_token = news_json.get("next_page_token", None)

        else:
            logger.error("Request failed with status code:", response.status_code)

        self._page_token = next_page_token

        return news_json["news"]
