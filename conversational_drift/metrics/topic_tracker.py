"""
Topic extraction, turn clustering, and dimensionality reduction for conversational trajectory.
"""

from typing import List, Dict, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "yeah", "yes", "okay", "ok", "hi", "hello", "thanks", "thank"
}


def extract_keywords_per_turn(texts: List[str], top_n: int = 4) -> List[List[str]]:
    """
    Extracts top discriminative keywords for each turn using TF-IDF.
    """
    if not texts:
        return []

    cleaned = [re.sub(r"[^\w\s]", " ", t.lower()) for t in texts]
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOPWORDS),
            max_df=0.95,
            min_df=1,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned)
        feature_names = np.array(vectorizer.get_feature_names_out())

        keywords_list = []
        for row in tfidf_matrix:
            dense_row = row.toarray().flatten()
            if dense_row.sum() == 0:
                keywords_list.append([])
                continue
            top_indices = np.argsort(dense_row)[::-1][:top_n]
            # filter zero weights
            valid_indices = [idx for idx in top_indices if dense_row[idx] > 0]
            keywords = [feature_names[i] for i in valid_indices]
            keywords_list.append(keywords)
        return keywords_list
    except Exception:
        # Fallback to simple frequency count if vectorizer fails (e.g. single word total)
        keywords_list = []
        for text in cleaned:
            words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
            keywords_list.append(words[:top_n])
        return keywords_list


def cluster_dialogue_turns(
    embeddings: np.ndarray,
    turn_texts: List[str],
    max_k: int = 4
) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Clusters turns into coherent thematic groupings and summarizes key terms per cluster.
    """
    n = len(embeddings)
    if n < 2:
        return [0] * n, {0: ["General Discussion"]}

    k = min(max_k, n // 2) if n > 4 else min(2, n)
    k = max(1, k)

    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
    except Exception:
        labels = np.zeros(n, dtype=int)

    # Summarize clusters with top terms
    cluster_summaries: Dict[int, List[str]] = {}
    for c_id in range(k):
        indices = np.where(labels == c_id)[0]
        if len(indices) == 0:
            cluster_summaries[c_id] = ["Empty Topic"]
            continue
        cluster_texts = [turn_texts[i] for i in indices]
        # Get top terms for this cluster
        top_kws = extract_keywords_per_turn([" ".join(cluster_texts)], top_n=5)
        cluster_summaries[c_id] = top_kws[0] if top_kws and top_kws[0] else [f"Topic {c_id}"]

    return labels.tolist(), cluster_summaries


def compute_2d_trajectory_coords(embeddings: np.ndarray) -> np.ndarray:
    """
    Projects embeddings down to 2 dimensions using PCA for visual trajectory mapping.
    Returns array of shape (N, 2).
    """
    n, dim = embeddings.shape
    if n == 0:
        return np.empty((0, 2))
    if n == 1:
        return np.zeros((1, 2))

    if n == 2:
        # Simple projection
        diff = embeddings[1] - embeddings[0]
        norm = np.linalg.norm(diff)
        return np.array([[0.0, 0.0], [norm, 0.0]])

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    return coords
