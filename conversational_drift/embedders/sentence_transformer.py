"""
Dense Transformer embedder using sentence-transformers (HuggingFace).
"""

from typing import List, Optional
import numpy as np
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    State-of-the-art dense semantic embedder using the `sentence-transformers` library.
    Default model: 'all-MiniLM-L6-v2' (fast, compact 384-dimensional dense vectors).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required for SentenceTransformerEmbedder. "
                "Install it with `pip install sentence-transformers` or use the fallback TFIDFEmbedder."
            ) from e

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Generate embeddings and normalize to unit hypersphere
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"SentenceTransformer({self.model_name})"
