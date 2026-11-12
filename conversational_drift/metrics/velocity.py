"""
Drift velocity, acceleration, and inflection point detectors.
"""

from typing import List, Tuple, Optional
import numpy as np
from ..models import DriftStatus, DriftZone


def compute_drift_kinematics(anchor_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes first derivative (velocity) and second derivative (acceleration)
    of anchor distance trajectory over conversational turns.

    V(t) = D(t) - D(t-1)
    A(t) = V(t) - V(t-1)
    """
    n = len(anchor_distances)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array([0.0]), np.array([0.0])

    velocity = np.zeros(n, dtype=np.float32)
    acceleration = np.zeros(n, dtype=np.float32)

    # First difference
    velocity[1:] = anchor_distances[1:] - anchor_distances[:-1]

    # Second difference
    if n > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]

    return velocity, acceleration


def classify_drift_status(
    anchor_distance: float,
    mild_threshold: float = 0.35,
    moderate_threshold: float = 0.60,
    severe_threshold: float = 0.80
) -> DriftStatus:
    """Classifies a turn into a discrete drift severity level based on semantic distance."""
    if anchor_distance < mild_threshold:
        return DriftStatus.ON_TOPIC
    elif anchor_distance < moderate_threshold:
        return DriftStatus.MILD_ELABORATION
    elif anchor_distance < severe_threshold:
        return DriftStatus.MODERATE_DRIFT
    else:
        return DriftStatus.SEVERE_DRIFT


def detect_inflection_points(
    anchor_distances: np.ndarray,
    velocities: np.ndarray,
    distance_threshold: float = 0.45,
    velocity_threshold: float = 0.18
) -> List[bool]:
    """
    Detects inflection points where the conversation abruptly pivots or breaks away from the topic.
    An inflection point occurs when:
    1. The distance jumps by more than `velocity_threshold` in a single turn, OR
    2. Distance exceeds `distance_threshold` while velocity is positive and greater than 0.10.
    """
    n = len(anchor_distances)
    inflections = [False] * n

    for i in range(1, n):
        d = anchor_distances[i]
        v = velocities[i]

        # Sudden sharp jump or cross-over with strong positive momentum
        if v >= velocity_threshold:
            inflections[i] = True
        elif d >= distance_threshold and v >= 0.10 and anchor_distances[i - 1] < distance_threshold:
            inflections[i] = True

    return inflections


def identify_drift_zones(
    anchor_distances: np.ndarray,
    statuses: List[DriftStatus],
    turns_texts: List[str],
    min_length: int = 2
) -> List[DriftZone]:
    """
    Identifies contiguous segments of turns where the conversation remains in MODERATE or SEVERE drift.
    """
    zones: List[DriftZone] = []
    in_drift = False
    start_idx = 0

    for i, status in enumerate(statuses):
        is_drifting = status in (DriftStatus.MODERATE_DRIFT, DriftStatus.SEVERE_DRIFT)

        if is_drifting and not in_drift:
            in_drift = True
            start_idx = i
        elif not is_drifting and in_drift:
            in_drift = False
            if (i - start_idx) >= min_length:
                seg_statuses = statuses[start_idx:i]
                has_severe = any(s == DriftStatus.SEVERE_DRIFT for s in seg_statuses)
                sev = DriftStatus.SEVERE_DRIFT if has_severe else DriftStatus.MODERATE_DRIFT
                zones.append(DriftZone(
                    start_turn=start_idx,
                    end_turn=i - 1,
                    severity=sev,
                    dominant_keywords=[],
                    description=f"Contiguous drift segment spanning turns {start_idx} to {i - 1}."
                ))

    # Catch tail zone
    if in_drift and (len(statuses) - start_idx) >= min_length:
        seg_statuses = statuses[start_idx:]
        has_severe = any(s == DriftStatus.SEVERE_DRIFT for s in seg_statuses)
        sev = DriftStatus.SEVERE_DRIFT if has_severe else DriftStatus.MODERATE_DRIFT
        zones.append(DriftZone(
            start_turn=start_idx,
            end_turn=len(statuses) - 1,
            severity=sev,
            dominant_keywords=[],
            description=f"Persistent drift continuing until conversation termination (turn {start_idx} to {len(statuses) - 1})."
        ))

    return zones
