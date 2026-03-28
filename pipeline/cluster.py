"""K-means clustering and category centroid computation for paper embeddings.

Provides two independent operations on the embedding matrix:
1. MiniBatchKMeans clustering (groups papers into k clusters).
2. Category centroid computation (mean embedding per arXiv category tag).

Both produce unit-norm output vectors.
"""

from __future__ import annotations

import numpy as np


def fit_kmeans(
    embeddings: np.ndarray,
    k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster paper embeddings using MiniBatchKMeans.

    Args:
        embeddings: Shape (N, 768), float32, unit-norm rows.
        k: Number of clusters. Default 500.

    Returns:
        Tuple of:
            cluster_ids: Shape (N,) int32 — cluster assignment for each paper.
            centroids: Shape (k, 768) float32 — unit-norm cluster centroids.

    Implementation:
        - Use MiniBatchKMeans(n_clusters=k, n_init=5, random_state=42, batch_size=4096).
        - Fit on embeddings.
        - Extract labels_ as cluster_ids (cast to int32).
        - Extract cluster_centers_ as centroids.
        - Normalize each centroid row to unit length before returning.
    """
    raise NotImplementedError


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

    Implementation:
        - For each unique tag across all papers, collect the indices of papers
          that have that tag.
        - Compute the mean of the corresponding embedding rows.
        - Normalize each mean vector to unit length.
        - Return as a dict.
    """
    raise NotImplementedError
