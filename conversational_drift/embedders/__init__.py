"""Embedding backends for conversational drift."""
from .base import BaseEmbedder
from .tfidf_embedder import TFIDFEmbedder
from .factory import get_embedder

__all__ = ["BaseEmbedder", "TFIDFEmbedder", "get_embedder"]
