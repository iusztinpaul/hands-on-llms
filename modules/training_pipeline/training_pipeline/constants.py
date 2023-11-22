from enum import Enum
from pathlib import Path


class Scope(Enum):
    """
    Enum class representing the different scopes used in the training pipeline.
    """

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "test"
    INFERENCE = "inference"


CACHE_DIR = Path.home() / ".cache" / "hands-on-llms"
