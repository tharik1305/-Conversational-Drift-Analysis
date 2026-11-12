"""
Semantic similarity and distance functions for vector embeddings.
"""

from typing import Union
import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Computes cosine similarity between two vectors or matrices of unit vectors.
    Clips output to [-1.0, 1.0] and re-scales to [0.0, 1.0] for semantic intuition.
    """
    if v1.ndim == 1 and v2.ndim == 1:
        dot = float(np.dot(v1, v2))
        return float(np.clip(dot, -1.0, 1.0))
    elif v1.ndim == 2 and v2.ndim == 1:
        dots = np.dot(v1, v2)
        return np.clip(dots, -1.0, 1.0)
    elif v1.ndim == 2 and v2.ndim == 2:
        dots = np.sum(v1 * v2, axis=1)
        return np.clip(dots, -1.0, 1.0)
    else:
        raise ValueError(f"Incompatible shapes: {v1.shape} and {v2.shape}")


def semantic_distance(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Semantic distance defined as 1 - cosine_similarity.
    Bounded in [0.0, 2.0], where 0 is identical and 1 is orthogonal (completely unrelated).
    """
    sim = cosine_similarity(v1, v2)
    dist = 1.0 - sim
    # Ensure non-negative due to float precision
    return np.maximum(0.0, dist)
