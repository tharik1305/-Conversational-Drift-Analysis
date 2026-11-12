"""
Tests for embedding backends.
"""

import numpy as np
import pytest
from conversational_drift.embedders.tfidf_embedder import TFIDFEmbedder
from conversational_drift.embedders.factory import get_embedder


def test_tfidf_embedder_normalization():
    texts = [
        "How do I reset my password?",
        "I need help with my account login.",
        "What is the weather like in Seattle today?",
        "Let's get some seafood chowder."
    ]
    embedder = TFIDFEmbedder(n_components=10)
    vectors = embedder.embed(texts)

    assert vectors.shape[0] == 4
    # Check L2 norms are equal to 1.0
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_tfidf_semantic_distance():
    texts = [
        "Reset password on account",
        "Help resetting forgotten password",
        "Exploring Seattle space needle tourist attractions"
    ]
    embedder = TFIDFEmbedder()
    vectors = embedder.embed(texts)

    # Similar queries should have higher cosine dot product
    sim_similar = float(np.dot(vectors[0], vectors[1]))
    sim_unrelated = float(np.dot(vectors[0], vectors[2]))

    assert sim_similar > sim_unrelated


def test_embedder_factory_auto():
    embedder = get_embedder(embedder_type="auto")
    assert embedder is not None
    assert hasattr(embedder, "embed")
