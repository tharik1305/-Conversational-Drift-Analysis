"""
Base interface for text embedding models.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all text embedder backends."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings into normalized vector representations.

        Args:
            texts: List of strings to embed.

        Returns:
            2D numpy array of shape (len(texts), dimension), normalized to unit length.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the embedding model."""
        pass

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalizes vectors so that dot products equal cosine similarities."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms
