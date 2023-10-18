from pathlib import Path

# == Embeddings model ==
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384

# == VECTOR Database ==
VECTOR_DB_OUTPUT_COLLECTION_NAME = "alpaca_financial_news"
VECTOR_DB_SEARCH_TOPK = 1

# == LLM Model ==
LLM_MODEL_ID = "tiiuae/falcon-7b-instruct"
LLM_QLORA_CHECKPOINT = "joywalker/financial-assistant-falcon-7b:1.0.0"

LLM_INFERNECE_MAX_NEW_TOKENS = 250
LLM_INFERENCE_TEMPERATURE = 1.0


# == Prompt Template ==
TEMPLATE_NAME = "falcon"
SYSTEM_MESSAGE = "You are a financial expert. Based on the context I provide, respond in a helpful manner"

# === Misc ===
CACHE_DIR = Path.home() / ".cache" / "hands-on-llms"
