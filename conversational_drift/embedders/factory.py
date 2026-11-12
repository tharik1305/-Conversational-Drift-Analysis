"""
Embedder factory with automatic backend selection and graceful fallbacks.
"""

import logging
from typing import Optional
from .base import BaseEmbedder
from .tfidf_embedder import TFIDFEmbedder

logger = logging.getLogger("conversational_drift.embedders")


def get_embedder(
    embedder_type: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None
) -> BaseEmbedder:
    """
    Factory function to instantiate an appropriate embedder.

    Args:
        embedder_type: 'auto', 'sentence-transformers', or 'tfidf'.
        model_name: HuggingFace model name if using sentence-transformers.
        device: 'cpu', 'cuda', etc.

    Returns:
        Instance of BaseEmbedder.
    """
    embedder_type = embedder_type.lower()

    if embedder_type in ("sentence-transformers", "transformer", "dense"):
        from .sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name=model_name, device=device)

    elif embedder_type in ("tfidf", "lsa", "sparse"):
        return TFIDFEmbedder()

    elif embedder_type == "auto":
        try:
            from .sentence_transformer import SentenceTransformerEmbedder
            logger.info(f"Loading dense neural embedder: {model_name}")
            return SentenceTransformerEmbedder(model_name=model_name, device=device)
        except (ImportError, Exception) as e:
            logger.warning(
                f"SentenceTransformer not available ({e}). Falling back to lightweight TF-IDF / LSA embedder."
            )
            return TFIDFEmbedder()

    else:
        raise ValueError(f"Unknown embedder type: '{embedder_type}'. Choose 'auto', 'sentence-transformers', or 'tfidf'.")
