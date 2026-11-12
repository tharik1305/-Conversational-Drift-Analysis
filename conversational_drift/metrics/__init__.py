"""
Metrics module for conversational drift.
"""

from .semantic_distance import cosine_similarity, semantic_distance
from .coherence import (
    compute_turn_to_turn_distances,
    compute_context_window_coherence,
    compute_conversation_cohesion_index,
)
from .velocity import (
    compute_drift_kinematics,
    classify_drift_status,
    detect_inflection_points,
    identify_drift_zones,
)
from .topic_tracker import (
    extract_keywords_per_turn,
    cluster_dialogue_turns,
    compute_2d_trajectory_coords,
)

__all__ = [
    "cosine_similarity",
    "semantic_distance",
    "compute_turn_to_turn_distances",
    "compute_context_window_coherence",
    "compute_conversation_cohesion_index",
    "compute_drift_kinematics",
    "classify_drift_status",
    "detect_inflection_points",
    "identify_drift_zones",
    "extract_keywords_per_turn",
    "cluster_dialogue_turns",
    "compute_2d_trajectory_coords",
]
