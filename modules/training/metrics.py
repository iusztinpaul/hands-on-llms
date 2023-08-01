import numpy as np

from transformers import EvalPrediction


def compute_perplexity(predictions: np.ndarray) -> float:
    """Compute perplexity metric."""

    return np.exp(predictions.mean()).item()
