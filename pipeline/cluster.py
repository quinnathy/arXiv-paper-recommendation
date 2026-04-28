"""K-means clustering and category centroid computation for paper embeddings.

Provides two independent operations on the embedding matrix:
1. MiniBatchKMeans clustering (groups papers into k clusters).
2. Category centroid computation (mean embedding per arXiv category tag).

Both produce unit-norm output vectors.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def fit_kmeans(
    embeddings: np.ndarray,
    k: int = 500,
    batch_size: int = 4096,
    max_iter: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster paper embeddings using MiniBatchKMeans.

    Args:
        embeddings: Shape (N, 768), float32, unit-norm rows.
        k: Number of clusters. Default 500.
        batch_size: MiniBatchKMeans batch size.
        max_iter: Maximum MiniBatchKMeans iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of:
            cluster_ids: Shape (N,) int32 — cluster assignment for each paper.
            centroids: Shape (k, 768) float32 — unit-norm cluster centroids.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        n_init=5,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
    )
    kmeans.fit(embeddings)

    cluster_ids = kmeans.labels_.astype(np.int32)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    # Normalize centroids to unit length
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    return cluster_ids, centroids


def compute_category_centroids(
    embeddings: np.ndarray,
    paper_meta: list[dict],
) -> dict[str, np.ndarray]:
    """Compute the mean embedding for each arXiv category tag.

    Args:
        embeddings: Shape (N, 768), float32, unit-norm rows.
        paper_meta: Length-N list of dicts. Each dict has a "categories" key
            containing a list of arXiv tags, e.g. ["cs.LG", "cs.CL"].

    Returns:
        Dict mapping category string to its unit-norm centroid (shape (768,), float32).
        Example: {"cs.LG": array(768,), "cs.CL": array(768,), ...}
    """
    # Collect paper indices per category tag
    cat_indices: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(paper_meta):
        for cat in meta["categories"]:
            cat_indices[cat].append(i)

    # Compute mean embedding per category, then normalize
    centroids: dict[str, np.ndarray] = {}
    for cat, indices in cat_indices.items():
        mean_vec = embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            mean_vec = mean_vec / norm
        centroids[cat] = mean_vec.astype(np.float32)

    return centroids
