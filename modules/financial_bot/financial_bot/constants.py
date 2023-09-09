from pathlib import Path

# == Embeddings model ==
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384
EMBEDDING_MODEL_DEVICE = "cuda:0"
VECTOR_DB_OUTPUT_COLLECTION_NAME = "test_collection"

# == LLM Model ==
LLM_MODEL_ID = ""
LLM_QLORA_CHECKPOINT = ""

CACHE_DIR = Path.home() / ".cache" / "hands-on-llms"
