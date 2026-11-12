"""
Fast, lightweight TF-IDF and Latent Semantic Analysis (LSA) Embedder.
Zero PyTorch dependency, runs instantaneously on any system.
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .base import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    """
    TF-IDF with Latent Semantic Analysis (LSA via TruncatedSVD).
    Transforms dialogue turns into dense, semantic latent vectors using vocabulary distribution.
    Useful for quick benchmarks, offline environments, and lightweight testing.
    """

    def __init__(self, n_components: int = 64, ngram_range: tuple = (1, 2)):
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            min_df=1
        )
        self.svd: TruncatedSVD = None
        self._fitted = False
        self._dim = n_components

    def fit(self, corpus: List[str]):
        """Fits the vocabulary and latent projection components on a conversational corpus."""
        if not corpus or len(corpus) == 0:
            return

        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        n_samples, n_features = tfidf_matrix.shape
        
        # SVD components cannot exceed min(n_samples, n_features) - 1
        usable_components = min(self.n_components, n_samples - 1, n_features - 1)
        if usable_components >= 2:
            self.svd = TruncatedSVD(n_components=usable_components, random_state=42)
            self.svd.fit(tfidf_matrix)
            self._dim = usable_components
        else:
            self.svd = None
            self._dim = n_features
            
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if not self._fitted:
            self.fit(texts)

        tfidf_matrix = self.vectorizer.transform(texts)
        
        if self.svd is not None:
            dense_vectors = self.svd.transform(tfidf_matrix)
        else:
            dense_vectors = tfidf_matrix.toarray()

        return self.normalize(dense_vectors.astype(np.float32))

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"TFIDF-LSA(dim={self._dim})"
