import logging
import os
from typing import Optional

import qdrant_client

logger = logging.getLogger(__name__)


def build_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Builds a Qdrant client object using the provided URL and API key.

    Args:
        url (Optional[str]): The URL of the Qdrant server. If not provided, the function will attempt
            to read it from the QDRANT_URL environment variable.
        api_key (Optional[str]): The API key to use for authentication. If not provided, the function will attempt
            to read it from the QDRANT_API_KEY environment variable.

    Raises:
        KeyError: If the URL or API key is not provided and cannot be read from the environment variables.

    Returns:
        qdrant_client.QdrantClient: A Qdrant client object.
    """

    logger.info("Building QDrant Client")
    if url is None:
        try:
            url = os.environ["QDRANT_URL"]
        except KeyError:
            raise KeyError(
                "QDRANT_URL must be set as environment variable or manually passed as an argument."
            )

    if api_key is None:
        try:
            api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
            )

    client = qdrant_client.QdrantClient(url, api_key=api_key)

    return client
