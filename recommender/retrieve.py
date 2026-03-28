"""Candidate retrieval: cluster selection + KNN search within clusters.

Two-stage retrieval:
1. Find the nearest clusters to the user embedding (cheap, operates on k centroids).
2. Brute-force KNN within those clusters (more expensive, but scoped to a subset).

All similarity computations use dot product (embeddings are unit-norm).
"""

from __future__ import annotations

import numpy as np

from pipeline.index import PaperIndex


def find_nearest_clusters(
    user_emb: np.ndarray,
    centroids: np.ndarray,
    n: int = 2,
) -> list[int]:
    """Find the n cluster centroids most similar to the user embedding.

    Args:
        user_emb: Shape (768,), float32, unit-norm.
        centroids: Shape (k, 768), float32, unit-norm rows.
        n: Number of top clusters to return.

    Returns:
        List of n cluster indices (ints), sorted by descending similarity.

    Implementation:
        - sims = centroids @ user_emb   → shape (k,)
        - Return indices of the top-n values in descending order.
    """
    raise NotImplementedError


def knn_in_clusters(
    user_emb: np.ndarray,
    target_cluster_ids: list[int],
    index: PaperIndex,
    seen_ids: set[str],
    k: int = 40,
) -> list[tuple[float, dict]]:
    """Find the k most similar papers within the specified clusters.

    Filters out papers the user has already seen.

    Args:
        user_emb: Shape (768,), float32, unit-norm.
        target_cluster_ids: List of cluster IDs to search within.
        index: The loaded PaperIndex containing all embeddings and metadata.
        seen_ids: Set of arXiv paper IDs to exclude (already seen by user).
        k: Maximum number of candidates to return.

    Returns:
        List of (similarity_score, paper_meta_dict) tuples, sorted by
        descending similarity. Length is min(k, available unseen papers).

    Implementation:
        - mask = np.isin(index.cluster_ids, target_cluster_ids)
        - Extract candidate embeddings: index.embeddings[mask]  → shape (M, 768)
        - Extract candidate metadata: [index.paper_meta[i] for i, b in enumerate(mask) if b]
        - Compute similarities: cand_embs @ user_emb  → shape (M,)
        - Sort descending.
        - Filter out papers whose "id" is in seen_ids.
        - Return top k as [(score, meta), ...].
    """
    raise NotImplementedError
