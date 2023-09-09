import logging
import os
from typing import Optional

import qdrant_client
from financial_bot import constants
from langchain.vectorstores import Qdrant

logger = logging.getLogger(__name__)


def build_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
):
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
    # retriever = Qdrant(client=client, collection_name=collection_name)

    return client
