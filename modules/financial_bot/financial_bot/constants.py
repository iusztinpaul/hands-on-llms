from pathlib import Path

# == Embeddings model ==
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384
EMBEDDING_MODEL_DEVICE = "cuda:0"
VECTOR_DB_OUTPUT_COLLECTION_NAME = "test_collection"

# == LLM Model ==
LLM_MODEL_ID = "tiiuae/falcon-7b-instruct"
LLM_QLORA_CHECKPOINT = "joywalker/financial-assistant-falcon-7b:1.0.0"
CACHE_DIR = Path.home() / ".cache" / "hands-on-llms"

# == Prompt Template ==
TEMPLATE_NAME = "falcon"
