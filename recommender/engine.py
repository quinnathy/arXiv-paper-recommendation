"""Top-level recommendation engine.

Provides a single recommend() entry point that orchestrates retrieval and
reranking to produce the final list of recommended papers.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

import numpy as np

from pipeline.index import PaperIndex
from recommender.retrieve import find_nearest_clusters, knn_in_clusters
from recommender.rerank import recency_score


logger = logging.getLogger(__name__)

TARGET_RECOMMENDATIONS = 5
INITIAL_TOP_M = 80
EXPANDED_TOP_M = 150
MAX_INITIAL_CLUSTERS = 8
MAX_EXPANDED_CLUSTERS = 16
RELAXED_CLUSTER_CAP = 2


def _normalize_cluster_ids(cluster_ids: Iterable[int] | np.ndarray) -> set[int]:
    """Normalize cluster IDs from numpy/list/set forms into plain ints."""
    return {int(cid) for cid in list(cluster_ids)}


def _rank_selected_clusters(
    index: PaperIndex,
    user_centroids: np.ndarray,
    cluster_ids: set[int],
) -> list[int]:
    """Rank a cluster subset by its best similarity to any user centroid."""
    if not cluster_ids:
        return []
    sims = index.centroids @ user_centroids.T
    max_sims = sims.max(axis=1)
    return sorted(cluster_ids, key=lambda cid: float(max_sims[cid]), reverse=True)


def _cap_selected_clusters(
    index: PaperIndex,
    user_centroids: np.ndarray,
    cluster_ids: Iterable[int] | np.ndarray,
    max_total_clusters: int,
) -> list[int]:
    """Keep an existing cluster selection under a hard cluster cap."""
    selected = _normalize_cluster_ids(cluster_ids)
    if len(selected) <= max_total_clusters:
        return sorted(selected)
    return sorted(
        _rank_selected_clusters(index, user_centroids, selected)[:max_total_clusters]
    )


def expand_clusters_near_user(
    index: PaperIndex,
    user_centroids: np.ndarray,
    selected_clusters: Iterable[int] | np.ndarray,
    max_total_clusters: int = MAX_EXPANDED_CLUSTERS,
) -> list[int]:
    """
    Expand selected retrieval clusters by adding the nearest k-means clusters
    to the user centroids, but never exceed max_total_clusters.
    """
    total_clusters = int(index.centroids.shape[0])
    if total_clusters <= 0:
        return []

    # Leave at least one cluster unscanned when possible, making the no-all-
    # clusters fallback invariant explicit even for small synthetic indexes.
    effective_cap = min(max_total_clusters, max(1, total_clusters - 1))
    expanded = _normalize_cluster_ids(selected_clusters)

    if len(expanded) > effective_cap:
        expanded = set(
            _rank_selected_clusters(index, user_centroids, expanded)[:effective_cap]
        )

    rankings: list[list[int]] = []
    for centroid in user_centroids:
        sims = index.centroids @ centroid
        rankings.append(np.argsort(sims)[::-1].astype(int).tolist())

    depth = 0
    while len(expanded) < effective_cap:
        added = False
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            cid = int(ranking[depth])
            if cid not in expanded:
                expanded.add(cid)
                added = True
                if len(expanded) >= effective_cap:
                    break
        if not added and all(depth >= len(ranking) for ranking in rankings):
            break
        depth += 1

    assert len(expanded) <= max_total_clusters
    if total_clusters > 1:
        assert len(expanded) < total_clusters
    return sorted(expanded)


def _score_candidates(
    candidates: list[tuple[float, dict, int]],
    recency_weight: float = 0.25,
    noise_scale: float = 0.02,
) -> list[tuple[float, dict, int]]:
    rng = np.random.default_rng()
    scored: list[tuple[float, dict, int]] = []
    for sim_score, meta, nearest_ci in candidates:
        bonus = recency_weight * recency_score(meta.get("update_date", ""))
        noise = rng.uniform(-noise_scale, noise_scale)
        final = sim_score + bonus + noise
        meta["rec_score"] = final
        scored.append((final, meta, nearest_ci))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def select_with_relaxation(
    candidates: list[tuple[float, dict, int]],
    k_u: int,
    diversity: float,
    n: int = TARGET_RECOMMENDATIONS,
    seen_ids: set[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Select recommendations using strict, relaxed, then score-only passes."""
    seen_ids = seen_ids or set()
    scored = _score_candidates(candidates)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    used_clusters: set[int] = set()
    covered_centroids: set[int] = set()

    for _score, meta, nearest_ci in scored:
        pid = meta.get("id")
        if pid in seen_ids or pid in selected_ids:
            continue
        cid = meta.get("cluster_id")
        if cid in used_clusters:
            continue
        if (
            diversity > 0.5
            and k_u > 1
            and nearest_ci in covered_centroids
            and len(covered_centroids) < k_u
        ):
            continue

        selected.append(meta)
        if pid is not None:
            selected_ids.add(pid)
        used_clusters.add(cid)
        covered_centroids.add(nearest_ci)
        if len(selected) >= n:
            return selected[:n], len(selected), len(selected)

    strict_count = len(selected)
    cluster_counts = Counter(meta.get("cluster_id") for meta in selected)

    for _score, meta, _nearest_ci in scored:
        pid = meta.get("id")
        if pid in seen_ids or pid in selected_ids:
            continue
        cid = meta.get("cluster_id")
        if cluster_counts[cid] >= RELAXED_CLUSTER_CAP:
            continue

        selected.append(meta)
        if pid is not None:
            selected_ids.add(pid)
        cluster_counts[cid] += 1
        if len(selected) >= n:
            return selected[:n], strict_count, len(selected)

    relaxed_count = len(selected)

    for _score, meta, _nearest_ci in scored:
        pid = meta.get("id")
        if pid in seen_ids or pid in selected_ids:
            continue

        selected.append(meta)
        if pid is not None:
            selected_ids.add(pid)
        if len(selected) >= n:
            break

    return selected[:n], strict_count, relaxed_count


def _dedupe_candidates(
    candidates: list[tuple[float, dict, int]],
    seen_ids: set[str],
) -> list[tuple[float, dict, int]]:
    """Keep one candidate per paper ID, preferring the highest raw similarity."""
    by_id: dict[str, tuple[float, dict, int]] = {}
    anonymous: list[tuple[float, dict, int]] = []

    for candidate in candidates:
        sim_score, meta, _nearest_ci = candidate
        pid = meta.get("id")
        if pid in seen_ids:
            continue
        if pid is None:
            anonymous.append(candidate)
            continue
        if pid not in by_id or sim_score > by_id[pid][0]:
            by_id[pid] = candidate

    deduped = list(by_id.values()) + anonymous
    deduped.sort(key=lambda item: item[0], reverse=True)
    return deduped


def recommend(
    user_centroids: np.ndarray,
    seen_ids: set[str],
    index: PaperIndex,
    diversity: float = 0.5,
    n: int = TARGET_RECOMMENDATIONS,
) -> list[dict]:
    """Generate n paper recommendations for a user.

    Args:
        user_centroids: Shape (k_u, 768), float32, unit-norm rows.
        seen_ids: Set of arXiv paper IDs already seen.
        index: Loaded PaperIndex.
        diversity: δ slider value, 0.0–1.0.
        n: Papers to recommend. Default 5.

    Returns:
        List of up to n paper_meta dicts with "rec_score" added.
    """
    k_u = user_centroids.shape[0]

    # 1. Cluster selection (δ controls budget)
    clusters = _cap_selected_clusters(
        index,
        user_centroids,
        find_nearest_clusters(user_centroids, index.centroids, diversity),
        MAX_INITIAL_CLUSTERS,
    )
    initial_clusters = list(clusters)

    # 2. KNN within those clusters
    candidates = knn_in_clusters(
        user_centroids,
        clusters,
        index,
        seen_ids,
        k=INITIAL_TOP_M,
    )

    # 3. Strict diversity filter, then relaxed passes on already-retrieved candidates
    results, strict_count, relaxed_count = select_with_relaxation(
        candidates,
        k_u=k_u,
        diversity=diversity,
        n=n,
        seen_ids=seen_ids,
    )

    bounded_expansion_used = False
    final_clusters = list(initial_clusters)
    if len(results) < n:
        expanded_clusters = expand_clusters_near_user(
            index=index,
            user_centroids=user_centroids,
            selected_clusters=initial_clusters,
            max_total_clusters=MAX_EXPANDED_CLUSTERS,
        )
        additional_clusters = sorted(set(expanded_clusters) - set(initial_clusters))
        if additional_clusters:
            bounded_expansion_used = True
            final_clusters = list(expanded_clusters)
            extra_candidates = knn_in_clusters(
                user_centroids,
                additional_clusters,
                index,
                seen_ids,
                k=EXPANDED_TOP_M,
            )
            candidates = _dedupe_candidates(candidates + extra_candidates, seen_ids)
            results, strict_count, relaxed_count = select_with_relaxation(
                candidates,
                k_u=k_u,
                diversity=diversity,
                n=n,
                seen_ids=seen_ids,
            )

    logger.info(
        "recommendations: initial_clusters=%d expanded_clusters=%d candidates=%d "
        "strict_selected=%d relaxed_selected=%d final_selected=%d "
        "bounded_expansion_used=%s",
        len(initial_clusters),
        len(final_clusters),
        len(candidates),
        strict_count,
        relaxed_count,
        len(results),
        bounded_expansion_used,
    )

    return results[:n]
