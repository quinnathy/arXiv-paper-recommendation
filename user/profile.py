"""User profile initialization and EMA-based centroid updates.

Handles two key operations:
1. Cold-start: construct a multi-vector user profile from weighted seed signals using threshold-based agglomerative grouping.
2. Feedback update: shift the nearest user centroid toward/away from a paper via EMA.

All output centroids are guaranteed unit-norm rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Feedback constants (unchanged)
# ---------------------------------------------------------------------------

FEEDBACK_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "save": 1.5,
    "skip": -0.3,
}

EMA_ALPHA: float = 0.15

# ---------------------------------------------------------------------------
# Agglomerative grouping defaults
# ---------------------------------------------------------------------------

MERGE_THRESHOLD: float = 0.22
MAX_THREADS: int = 3
CORE_SPLIT_POWER: float = 0.6


# ---------------------------------------------------------------------------
# SeedSignal
# ---------------------------------------------------------------------------

@dataclass
class SeedSignal:
    """A single onboarding seed that contributes to user centroid construction.

    Attributes:
        vector: Unit-norm (768,) embedding.
        weight: Importance weight (higher → more influence on the centroid).
        reliability: 0–1, how trustworthy this signal source is.
        specificity: 0–1, how narrow/specific the topic is.
        split_power: 0–1, ability to create its own research thread.
            Seeds below :data:`CORE_SPLIT_POWER` are treated as *support*
            seeds that attach to the nearest core thread.
        label: Human-readable label used for thread naming.
        source: Origin of this seed for diagnostics.
    """

    vector: np.ndarray
    weight: float
    reliability: float
    specificity: float
    split_power: float
    label: str
    source: Literal[
        "arxiv_category",
        "predefined_tag",
        "free_text",
        "scholar_title",
    ] = "arxiv_category"


@dataclass
class ProfileInitializationResult:
    """Result of threshold-based profile initialization.

    Attributes:
        centroids: Unit-norm user centroids, shape (k_u, 768).
        seed_labels: Integer thread assignment for each input seed,
            shape (n_seeds,).
        thread_weights: Normalized per-thread weights, shape (k_u,).
        thread_labels: Human-readable label for each thread.
    """

    centroids: np.ndarray
    seed_labels: np.ndarray
    thread_weights: np.ndarray
    thread_labels: list[str]


# ---------------------------------------------------------------------------
# Seed factory helpers
# ---------------------------------------------------------------------------

def make_category_seed(code: str, label: str, embedding: np.ndarray) -> SeedSignal:
    """Create a seed from an arXiv category centroid."""
    return SeedSignal(
        vector=embedding,
        weight=1.0,
        reliability=0.9,
        specificity=0.3,
        split_power=0.4,
        label=label,
        source="arxiv_category",
    )


def make_concept_seed(
    key: str, label: str, embedding: np.ndarray, broad: bool = False,
) -> SeedSignal:
    """Create a seed from a predefined concept tag."""
    if broad:
        return SeedSignal(
            vector=embedding,
            weight=1.0,
            reliability=0.9,
            specificity=0.5,
            split_power=0.3,
            label=label,
            source="predefined_tag",
        )
    return SeedSignal(
        vector=embedding,
        weight=1.5,
        reliability=0.9,
        specificity=0.7,
        split_power=0.7,
        label=label,
        source="predefined_tag",
    )


def make_scholar_seed(title: str, embedding: np.ndarray) -> SeedSignal:
    """Create a seed from a Google Scholar paper embedding."""
    return SeedSignal(
        vector=embedding,
        weight=1.5,
        reliability=0.95,
        specificity=0.8,
        split_power=0.9,
        label=title,
        source="scholar_title",
    )


def make_freetext_seed(phrase: str, embedding: np.ndarray) -> SeedSignal:
    """Create a seed from a user-provided free-text interest."""
    return SeedSignal(
        vector=embedding,
        weight=2.0,
        reliability=0.8,
        specificity=0.9,
        split_power=1.0,
        label=phrase,
        source="free_text",
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _effective_weight(seed: SeedSignal) -> float:
    return seed.weight * seed.reliability * seed.specificity


# ---------------------------------------------------------------------------
# Agglomerative threshold grouping
# ---------------------------------------------------------------------------

def _threshold_agglomerative_grouping(
    seeds: list[SeedSignal],
    merge_threshold: float = MERGE_THRESHOLD,
    max_threads: int = MAX_THREADS,
) -> list[list[int]]:
    """Group seed indices via agglomerative merging on cosine distance.

    Returns a list of groups, where each group is a list of indices into
    *seeds*.
    """
    groups: list[dict] = [
        {
            "indices": [i],
            "vectors": [s.vector],
            "weights": [_effective_weight(s)],
        }
        for i, s in enumerate(seeds)
    ]

    def _centroid(g: dict) -> np.ndarray:
        X = np.stack(g["vectors"])
        w = np.array(g["weights"])
        return _normalize((X * w[:, None]).sum(axis=0))

    def _dist(ga: dict, gb: dict) -> float:
        return 1.0 - float(_centroid(ga) @ _centroid(gb))

    # Phase 1: merge groups closer than threshold.
    while len(groups) > 1:
        best_pair = None
        best_dist = float("inf")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = _dist(groups[i], groups[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        if best_pair is None or best_dist > merge_threshold:
            break
        i, j = best_pair
        groups[i]["indices"].extend(groups[j]["indices"])
        groups[i]["vectors"].extend(groups[j]["vectors"])
        groups[i]["weights"].extend(groups[j]["weights"])
        del groups[j]

    # Phase 2: enforce max_threads.
    while len(groups) > max_threads:
        best_pair = None
        best_dist = float("inf")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = _dist(groups[i], groups[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        groups[i]["indices"].extend(groups[j]["indices"])
        groups[i]["vectors"].extend(groups[j]["vectors"])
        groups[i]["weights"].extend(groups[j]["weights"])
        del groups[j]

    return [g["indices"] for g in groups]


# ---------------------------------------------------------------------------
# Main initializer (v2)
# ---------------------------------------------------------------------------

def init_user_profile_v2(
    seeds: list[SeedSignal],
    max_threads: int = MAX_THREADS,
    merge_threshold: float = MERGE_THRESHOLD,
    core_split_power: float = CORE_SPLIT_POWER,
) -> ProfileInitializationResult:
    """Initialize user centroids from weighted seed signals.

    This is the replacement for KMeans-based initialization.
    The algorithm:

    1. Separates *core* seeds (high ``split_power``) from *support* seeds.
    2. Infers thread groups via agglomerative threshold grouping on core seeds.
    3. Assigns support seeds to the nearest inferred thread.
    4. Computes final weighted centroids.

    Args:
        seeds: List of :class:`SeedSignal` objects with unit-norm vectors.
        max_threads: Maximum number of user centroids.
        merge_threshold: Cosine distance threshold for merging groups.
        core_split_power: Minimum ``split_power`` to be a core seed.

    Returns:
        ProfileInitializationResult with centroids, per-seed integer thread
        assignments, thread labels, and normalized thread weights.
    """
    if len(seeds) == 0:
        raise ValueError("Cannot initialize user profile with zero seeds.")

    if len(seeds) == 1:
        s = seeds[0]
        return ProfileInitializationResult(
            centroids=s.vector.astype(np.float32).reshape(1, -1),
            seed_labels=np.array([0], dtype=int),
            thread_weights=np.array([1.0]),
            thread_labels=[s.label],
        )

    # 1. Partition core vs support.
    core_idx = [i for i, s in enumerate(seeds) if s.split_power >= core_split_power]
    support_idx = [i for i, s in enumerate(seeds) if s.split_power < core_split_power]

    # Fallback: if no core seeds, treat all as core.
    if not core_idx:
        core_idx = list(range(len(seeds)))
        support_idx = []

    core_seeds = [seeds[i] for i in core_idx]

    # 2. Agglomerative grouping on core seeds only.
    core_groups = _threshold_agglomerative_grouping(
        core_seeds,
        merge_threshold=merge_threshold,
        max_threads=max_threads,
    )

    # Build group centroids from core seeds for support-seed assignment.
    all_vectors = np.stack([s.vector for s in seeds])
    all_weights = np.array([_effective_weight(s) for s in seeds])

    # Map core-local indices back to global indices.
    global_groups: list[list[int]] = [
        [core_idx[li] for li in grp] for grp in core_groups
    ]

    # Compute core-group centroids for nearest-neighbour assignment.
    group_centroids = []
    for grp in global_groups:
        vecs = all_vectors[grp]
        wts = all_weights[grp]
        group_centroids.append(_normalize((vecs * wts[:, None]).sum(axis=0)))
    group_centroids_arr = np.stack(group_centroids)  # (n_groups, 768)

    # 3. Assign support seeds to nearest core group.
    for si in support_idx:
        sims = all_vectors[si] @ group_centroids_arr.T
        best_group = int(np.argmax(sims))
        global_groups[best_group].append(si)

    # 4. Compute final weighted centroids, labels, and weights.
    centroids = []
    thread_labels: list[str] = []
    thread_weights_raw: list[float] = []
    seed_labels = np.empty(len(seeds), dtype=int)

    for group_id, grp in enumerate(global_groups):
        vecs = all_vectors[grp]
        wts = all_weights[grp]
        c = _normalize((vecs * wts[:, None]).sum(axis=0)).astype(np.float32)
        centroids.append(c)

        # Label: highest-specificity seed in this group.
        best_label_idx = max(grp, key=lambda i: seeds[i].specificity)
        thread_labels.append(seeds[best_label_idx].label)

        thread_weights_raw.append(float(wts.sum()))
        for seed_idx in grp:
            seed_labels[seed_idx] = group_id

    centroids_arr = _normalize_rows(np.stack(centroids)).astype(np.float32)

    tw = np.array(thread_weights_raw)
    tw = tw / tw.sum()

    return ProfileInitializationResult(
        centroids=centroids_arr,
        seed_labels=seed_labels,
        thread_weights=tw,
        thread_labels=thread_labels,
    )


# ---------------------------------------------------------------------------
# Legacy wrapper (backward-compatible)
# ---------------------------------------------------------------------------

def init_user_profile(
    topic_keys: list[str],
    category_centroids: dict[str, np.ndarray],
    paper_embeddings: np.ndarray | None = None,
    max_k: int = 3,
) -> np.ndarray:
    """Initialize a multi-vector user profile from topics and optional papers.

    This is a backward-compatible wrapper around :func:`init_user_profile_v2`.
    It converts the old-style arguments into :class:`SeedSignal` objects and
    delegates to the new algorithm.

    Args:
        topic_keys: Selected arXiv category strings, e.g. ``["cs.LG", "cs.CL"]``.
        category_centroids: Dict mapping category string to unit-norm centroid (768,).
        paper_embeddings: Optional (n_papers, 768) float32 unit-norm embeddings
            from a Scholar or GitHub profile upload. ``None`` if tags only.
        max_k: Maximum number of user centroids. Default 3.

    Returns:
        Unit-norm centroids of shape ``(k_u, 768)``.
    """
    seeds: list[SeedSignal] = []

    for key in topic_keys:
        if key in category_centroids:
            seeds.append(make_category_seed(key, key, category_centroids[key]))

    if paper_embeddings is not None:
        for i in range(len(paper_embeddings)):
            seeds.append(make_scholar_seed(f"paper_{i}", paper_embeddings[i]))

    if not seeds:
        fallback = next(iter(category_centroids.values()))
        return fallback.astype(np.float32).copy().reshape(1, 768)

    result = init_user_profile_v2(seeds, max_threads=max_k)
    return result.centroids


# ---------------------------------------------------------------------------
# Feedback (unchanged)
# ---------------------------------------------------------------------------

def apply_feedback(
    centroids: np.ndarray,
    paper_embedding: np.ndarray,
    signal: str,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """Update the nearest user centroid via EMA after a feedback event.

    Only the centroid closest to the paper is modified. All other
    centroids remain unchanged, preserving distinct research threads.

    Args:
        centroids: Shape (k_u, 768), float32, unit-norm rows.
        paper_embedding: Shape (768,), float32, unit-norm.
        signal: One of "like", "save", "skip".
        alpha: EMA smoothing factor. Default 0.15.

    Returns:
        Updated centroids, same shape. The modified row is re-normalized.
        If the update would produce a degenerate vector (norm < 1e-8),
        the original centroids are returned unchanged.
    """
    w = FEEDBACK_WEIGHTS[signal]
    i_star = int(np.argmax(centroids @ paper_embedding))

    raw = (1 - alpha) * centroids[i_star] + alpha * w * paper_embedding
    norm = np.linalg.norm(raw)
    if norm < 1e-8:
        return centroids

    updated = centroids.copy()
    updated[i_star] = (raw / norm).astype(np.float32)
    return updated
