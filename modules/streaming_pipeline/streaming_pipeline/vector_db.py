import os

from qdrant_client import QdrantClient

from streaming_pipeline import initialize
from streaming_pipeline.embeddings import model, tokenizer

initialize()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


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
