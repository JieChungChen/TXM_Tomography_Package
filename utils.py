import numpy as np


def min_max_normalize(arr, max_percentile=100, min_percentile=0, to_8bit=False):
    """Min-max normalize an array based on given percentiles."""
    max_val = np.percentile(arr, max_percentile)
    min_val = np.percentile(arr, min_percentile)
    normalized = (arr - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    if to_8bit:
        normalized = (normalized * 255).astype(np.uint8)
    return normalized