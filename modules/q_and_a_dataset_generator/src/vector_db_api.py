import os
from typing import Dict, List

QDRANT_API_URL = os.environ['QDRANT_API_URL']
QDRANT_API_KEY = os.environ['QDRANT_API_KEY']

# def fetch_relevant_news_from_db() -> List[Dict]:
#     """"""
#     from qdrant_client import QdrantClient

#     qdrant_client = QdrantClient(
#         url="https://ae87052d-4765-49f5-b462-d9b468a97976.eu-central-1-0.aws.cloud.qdrant.io:6333", 
#         api_key="<your-token>",
#     )


from qdrant_client import QdrantClient

def get_qdrant_client() -> QdrantClient:
    """"""
    qdrant_client = QdrantClient(
        url=QDRANT_API_URL, 
        api_key=QDRANT_API_KEY,
    )

    return qdrant_client

def init_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int,
    # schema: str = ''
) -> QdrantClient:
    """"""
    from qdrant_client.http.api_client import UnexpectedResponse
    from qdrant_client.http.models import Distance, VectorParams

    try: 
        qdrant_client.get_collection(collection_name=collection_name)

    except (UnexpectedResponse, ValueError):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            # schema=schema
    )

    return qdrant_client
