"""
Tests for semantic distance, coherence, and drift kinematics.
"""

import numpy as np
import pytest
from conversational_drift.metrics.semantic_distance import cosine_similarity, semantic_distance
from conversational_drift.metrics.coherence import (
    compute_turn_to_turn_distances,
    compute_context_window_coherence,
    compute_conversation_cohesion_index,
)
from conversational_drift.metrics.velocity import (
    compute_drift_kinematics,
    classify_drift_status,
    detect_inflection_points,
    identify_drift_zones,
)
from conversational_drift.models import DriftStatus


def test_cosine_similarity_and_distance():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0], dtype=np.float32)

    assert pytest.approx(cosine_similarity(v1, v2), 1e-4) == 1.0
    assert pytest.approx(semantic_distance(v1, v2), 1e-4) == 0.0

    assert pytest.approx(cosine_similarity(v1, v3), 1e-4) == 0.0
    assert pytest.approx(semantic_distance(v1, v3), 1e-4) == 1.0


def test_turn_to_turn_distances():
    embs = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    sims, dists = compute_turn_to_turn_distances(embs)
    assert len(sims) == 3
    assert pytest.approx(sims[0]) == 1.0
    assert pytest.approx(sims[1]) == 1.0
    assert pytest.approx(sims[2]) == 0.0


def test_drift_kinematics():
    # Distances monotonically increasing: 0.0 -> 0.2 -> 0.6 -> 0.7
    dists = np.array([0.0, 0.2, 0.6, 0.7], dtype=np.float32)
    vel, acc = compute_drift_kinematics(dists)

    assert pytest.approx(vel[0]) == 0.0
    assert pytest.approx(vel[1]) == 0.2
    assert pytest.approx(vel[2]) == 0.4
    assert pytest.approx(vel[3]) == 0.1

    assert pytest.approx(acc[0]) == 0.0
    assert pytest.approx(acc[1]) == 0.0
    assert pytest.approx(acc[2]) == 0.2
    assert pytest.approx(acc[3]) == -0.3


def test_classify_drift_status():
    assert classify_drift_status(0.15) == DriftStatus.ON_TOPIC
    assert classify_drift_status(0.45) == DriftStatus.MILD_ELABORATION
    assert classify_drift_status(0.68) == DriftStatus.MODERATE_DRIFT
    assert classify_drift_status(0.92) == DriftStatus.SEVERE_DRIFT


def test_detect_inflection_points():
    dists = np.array([0.05, 0.10, 0.12, 0.75, 0.80], dtype=np.float32)
    vels = np.array([0.0, 0.05, 0.02, 0.63, 0.05], dtype=np.float32)

    inflections = detect_inflection_points(dists, vels, distance_threshold=0.35, velocity_threshold=0.18)
    assert inflections[3] is True
    assert inflections[0] is False
    assert inflections[1] is False
