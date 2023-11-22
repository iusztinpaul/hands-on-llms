import numpy as np


def compute_perplexity(predictions: np.ndarray) -> float:
    """
    Compute perplexity metric.

    Parameters:
    predictions (np.ndarray): Array of predicted values.

    Returns:
    float: Perplexity metric value.
    """

    return np.exp(predictions.mean()).item()
