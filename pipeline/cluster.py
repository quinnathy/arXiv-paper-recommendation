"""K-means clustering and category centroid computation for paper embeddings.

Provides two independent operations on the embedding matrix:
1. From-scratch MiniBatch K-means clustering for anonymous retrieval buckets.
2. Category centroid computation as labeled cold-start seed vectors.

Both produce unit-norm output vectors. The k-means clusters are spatial index
buckets for retrieval, not human-readable topics.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors."""
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, eps)).astype(np.float32, copy=False)


def _assign(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Assign each embedding to the nearest unit-norm centroid.

    For unit-norm vectors, nearest Euclidean centroid is equivalent to maximum
    dot product. Work is chunked to avoid materializing an ``N x K`` matrix for
    the full corpus.
    """
    labels = np.empty(int(embeddings.shape[0]), dtype=np.int32)
    for start in range(0, len(embeddings), chunk_size):
        end = min(start + chunk_size, len(embeddings))
        sims = embeddings[start:end] @ centroids.T
        labels[start:end] = np.argmax(sims, axis=1).astype(np.int32)
    return labels


def _kmeans_plus_plus_init(
    embeddings: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize centroids with k-means++ over unit-norm embeddings."""
    n = int(embeddings.shape[0])
    first = int(rng.integers(n))
    centroids = [np.asarray(embeddings[first], dtype=np.float32)]

    min_sq_dist = np.maximum(
        2.0 - 2.0 * (embeddings @ centroids[0]),
        0.0,
    ).astype(np.float64)

    for _ in range(1, k):
        total = float(min_sq_dist.sum())
        if total <= 1e-12 or not np.isfinite(total):
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=min_sq_dist / total))
        new_centroid = np.asarray(embeddings[idx], dtype=np.float32)
        centroids.append(new_centroid)

        sq_dist = np.maximum(2.0 - 2.0 * (embeddings @ new_centroid), 0.0)
        min_sq_dist = np.minimum(min_sq_dist, sq_dist)

    return _normalize_rows(np.stack(centroids))


def kmeans_inertia(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    chunk_size: int = 4096,
) -> float:
    """Compute sum of squared distances to assigned centroids."""
    total = 0.0
    labels = np.asarray(labels, dtype=np.int64)
    for start in range(0, len(embeddings), chunk_size):
        end = min(start + chunk_size, len(embeddings))
        batch = embeddings[start:end]
        batch_centroids = centroids[labels[start:end]]
        dots = np.sum(batch * batch_centroids, axis=1)
        total += float(np.maximum(2.0 - 2.0 * dots, 0.0).sum())
    return total


def fit_kmeans(
    embeddings: np.ndarray,
    k: int = 500,
    batch_size: int = 4096,
    max_iter: int = 20,
    random_state: int = 42,
    n_init: int = 5,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster paper embeddings using from-scratch MiniBatch K-means.

    Runs ``n_init`` independent k-means++ initializations and keeps the run with
    lowest final inertia. Each run performs up to ``max_iter`` shuffled passes
    over the data, updating centroids from mini-batch means with a decaying
    per-centroid learning rate.

    Args:
        embeddings: Shape ``(N, D)``, float32, expected unit-norm rows.
        k: Number of anonymous retrieval buckets.
        batch_size: Mini-batch size per update step.
        max_iter: Maximum shuffled passes over the data per initialization.
        random_state: Base random seed.
        n_init: Number of independent initializations.
        tol: Stop a run early when mean centroid shift falls below this value.

    Returns:
        ``(cluster_ids, centroids)`` where cluster IDs have shape ``(N,)`` and
        centroids have shape ``(k, D)`` with unit-norm rows.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}.")
    n, _d = embeddings.shape
    if n == 0:
        raise ValueError("Cannot cluster an empty embedding matrix.")
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, N], got k={k}, N={n}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if n_init <= 0:
        raise ValueError("n_init must be positive.")

    best_centroids: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")

    for run in range(n_init):
        rng = np.random.default_rng(random_state + run)
        centroids = _kmeans_plus_plus_init(embeddings, k, rng)
        counts = np.zeros(k, dtype=np.float64)

        for _epoch in range(max_iter):
            permutation = rng.permutation(n)
            old_centroids = centroids.copy()

            for start in range(0, n, batch_size):
                batch = embeddings[permutation[start : start + batch_size]]
                labels = _assign(batch, centroids, chunk_size=batch_size)

                for cluster_id in np.unique(labels):
                    mask = labels == cluster_id
                    assigned_count = int(mask.sum())
                    if assigned_count == 0:
                        continue
                    counts[cluster_id] += assigned_count
                    learning_rate = assigned_count / counts[cluster_id]
                    batch_mean = np.asarray(batch[mask].mean(axis=0), dtype=np.float32)
                    centroids[cluster_id] = (
                        (1.0 - learning_rate) * centroids[cluster_id]
                        + learning_rate * batch_mean
                    )

                centroids = _normalize_rows(centroids)

            shift = float(np.linalg.norm(centroids - old_centroids, axis=1).mean())
            if shift < tol:
                break

        labels = _assign(embeddings, centroids, chunk_size=batch_size)
        inertia = kmeans_inertia(embeddings, labels, centroids, chunk_size=batch_size)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    if best_labels is None or best_centroids is None:
        raise RuntimeError("K-means failed to produce cluster assignments.")
    return best_labels.astype(np.int32), _normalize_rows(best_centroids)


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
    cat_indices: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(paper_meta):
        for cat in meta["categories"]:
            cat_indices[cat].append(i)

    centroids: dict[str, np.ndarray] = {}
    for cat, indices in cat_indices.items():
        mean_vec = embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            mean_vec = mean_vec / norm
        centroids[cat] = mean_vec.astype(np.float32)

    return centroids
