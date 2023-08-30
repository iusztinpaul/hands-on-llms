import os
from typing import Optional

from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from streaming_pipeline.embeddings import model, tokenizer
from streaming_pipeline.models import Document


class _QdrantVectorSink(StatelessSink):
    def __init__(self, client: QdrantClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def write(self, document: Document):
        
        # TODO: Understand how the payloads are loaded in qdrant
        payloads = document.to_payloads()
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(id=idx, vector=vector, payload=_payload)
                for idx, (vector, _payload) in enumerate(zip(document.embeddings, payloads))
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
        collection_name: str,
        vector_size: int,
        # TODO: How can I use the schema ?
        schema="",
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.schema = schema

        try:
            self._qdrant_url = os.environ["QDRANT_URL"]
        except KeyError:
            raise KeyError("QDRANT_URL must be set as environment variable.")

        try:
            self._qdrant_api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError("QDRANT_API_KEY must be set as environment variable.")

        if client:
            self.client = client

        else:
            self.client = QdrantClient(self._qdrant_url, api_key=self._qdrant_api_key)

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


if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    query_string = "what is new on TSLA"
    inputs = tokenizer(
        query_string, padding=True, truncation=True, return_tensors="pt", max_length=384
    ).to("cuda:0")
    result = model(**inputs)
    embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
    lst = embeddings.flatten().tolist()

    query_vector = lst
    hits = client.search(
        collection_name="test_collection",
        query_vector=query_vector,
        limit=5,  # Return 5 closest points
    )
    print(hits)
