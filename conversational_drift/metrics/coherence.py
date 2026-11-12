"""
Coherence metrics for local and global conversational trajectory.
"""

from typing import List, Tuple
import numpy as np
from .semantic_distance import cosine_similarity, semantic_distance


def compute_turn_to_turn_distances(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes step-wise similarity and distance between consecutive turns (T_i vs T_{i-1}).
    For turn 0, stepwise similarity is set to 1.0 (distance 0.0).

    Returns:
        (similarities, distances) arrays of shape (N,)
    """
    n = len(embeddings)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array([1.0]), np.array([0.0])

    sims = np.ones(n, dtype=np.float32)
    dists = np.zeros(n, dtype=np.float32)

    # Dot product between adjacent pairs
    dot_products = np.sum(embeddings[1:] * embeddings[:-1], axis=1)
    sims[1:] = np.clip(dot_products, -1.0, 1.0)
    dists[1:] = np.maximum(0.0, 1.0 - sims[1:])

    return sims, dists


def compute_context_window_coherence(
    embeddings: np.ndarray,
    window_size: int = 3,
    decay: float = 0.8
) -> np.ndarray:
    """
    Computes coherence of turn T_i against a weighted moving context window of the preceding k turns.
    Weights decay exponentially for older turns in the window.

    Returns:
        Array of shape (N,) containing context window similarities in [0, 1].
    """
    n = len(embeddings)
    if n == 0:
        return np.array([])

    context_sims = np.ones(n, dtype=np.float32)

    for i in range(1, n):
        start_idx = max(0, i - window_size)
        window_embs = embeddings[start_idx:i]
        
        # Exponential weights for window: more recent turns have higher weight
        window_len = len(window_embs)
        weights = np.array([decay ** (window_len - 1 - j) for j in range(window_len)], dtype=np.float32)
        weights /= np.sum(weights)

        # Weighted centroid of previous context
        context_vec = np.sum(window_embs * weights[:, np.newaxis], axis=0)
        norm = np.linalg.norm(context_vec)
        if norm > 0:
            context_vec /= norm

        sim = float(np.dot(embeddings[i], context_vec))
        context_sims[i] = float(np.clip(sim, -1.0, 1.0))

    return context_sims


def compute_conversation_cohesion_index(anchor_distances: np.ndarray, stepwise_distances: np.ndarray) -> float:
    """
    Calculates the overall Conversation Cohesion Index (CCI) from 0.0 to 100.0%.
    Combines anchor fidelity (60% weight) and conversational flow / local coherence (40% weight).
    """
    if len(anchor_distances) == 0:
        return 100.0

    # Exclude turn 0 if it's the anchor itself
    relevant_anchor_dists = anchor_distances[1:] if len(anchor_distances) > 1 else anchor_distances
    relevant_step_dists = stepwise_distances[1:] if len(stepwise_distances) > 1 else stepwise_distances

    avg_anchor_dist = float(np.mean(relevant_anchor_dists)) if len(relevant_anchor_dists) > 0 else 0.0
    avg_step_dist = float(np.mean(relevant_step_dists)) if len(relevant_step_dists) > 0 else 0.0

    # Distance ranges usually between 0.0 (identical) and 1.0+ (completely unrelated)
    # Penalty is computed as a weighted loss
    weighted_penalty = 0.60 * avg_anchor_dist + 0.40 * avg_step_dist
    
    # Scale: penalty 0.0 -> 100%, penalty >= 1.0 -> 0%
    cohesion_score = max(0.0, min(100.0, (1.0 - weighted_penalty) * 100.0))
    return round(cohesion_score, 2)
