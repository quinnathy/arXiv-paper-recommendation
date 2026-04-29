from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import recommender.engine as engine


def _index(total_clusters: int = 40) -> SimpleNamespace:
    centroids = np.stack(
        [
            np.array([1.0 - i * 0.01, 0.0], dtype=np.float32)
            for i in range(total_clusters)
        ]
    )
    return SimpleNamespace(centroids=centroids)


def _candidate(
    paper_id: str,
    cluster_id: int,
    score: float,
    nearest_centroid: int = 0,
) -> tuple[float, dict, int]:
    return (
        score,
        {
            "id": paper_id,
            "title": paper_id,
            "abstract": "",
            "update_date": "2026-01-01",
            "cluster_id": cluster_id,
        },
        nearest_centroid,
    )


def _install_retrieval(
    monkeypatch,
    candidates_by_cluster: dict[int, list[tuple[float, dict, int]]],
    initial_clusters: list[int] | None = None,
) -> list[list[int]]:
    calls: list[list[int]] = []
    monkeypatch.setattr(
        engine,
        "find_nearest_clusters",
        lambda user_centroids, index_centroids, diversity: initial_clusters or [0],
    )

    def fake_knn(user_centroids, target_cluster_ids, index, seen_ids, k=engine.INITIAL_TOP_M):
        clusters = [int(cid) for cid in target_cluster_ids]
        calls.append(clusters)
        results: list[tuple[float, dict, int]] = []
        for cid in clusters:
            results.extend(candidates_by_cluster.get(cid, []))
        return [cand for cand in results if cand[1]["id"] not in seen_ids][:k]

    monkeypatch.setattr(engine, "knn_in_clusters", fake_knn)
    return calls


def test_no_all_cluster_fallback(monkeypatch):
    calls = _install_retrieval(
        monkeypatch,
        {
            0: [_candidate("p0", 0, 1.0)],
            1: [_candidate("p1", 1, 0.9)],
        },
    )

    recs = engine.recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_index(total_clusters=40),
        n=engine.TARGET_RECOMMENDATIONS,
    )

    all_clusters = set(range(40))
    scanned = set().union(*(set(call) for call in calls))
    assert len(recs) < engine.TARGET_RECOMMENDATIONS
    assert len(scanned) <= engine.MAX_EXPANDED_CLUSTERS
    assert scanned != all_clusters


def _cluster_candidates(
    cluster_count: int,
    per_cluster: int = 2,
) -> dict[int, list[tuple[float, dict, int]]]:
    candidates: dict[int, list[tuple[float, dict, int]]] = {}
    score = 1.0
    for cid in range(cluster_count):
        candidates[cid] = []
        for slot in range(per_cluster):
            candidates[cid].append(_candidate(f"p{cid}-{slot}", cid, score))
            score -= 0.001
    return candidates


def test_cluster_cap_happens_before_expansion(monkeypatch):
    calls = _install_retrieval(
        monkeypatch,
        _cluster_candidates(10, per_cluster=2),
        initial_clusters=list(range(10)),
    )

    recs = engine.recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_index(total_clusters=40),
        n=engine.TARGET_RECOMMENDATIONS,
    )

    assert len(recs) == engine.TARGET_RECOMMENDATIONS
    assert len(calls) == 1


def test_bounded_expansion_is_used_when_initial_candidates_are_insufficient(monkeypatch):
    calls = _install_retrieval(
        monkeypatch,
        _cluster_candidates(20, per_cluster=2),
    )

    recs = engine.recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_index(total_clusters=40),
        n=engine.TARGET_RECOMMENDATIONS,
    )

    scanned = set().union(*(set(call) for call in calls))
    assert len(recs) == engine.TARGET_RECOMMENDATIONS
    assert len(calls) == 2
    assert len(scanned) <= engine.MAX_EXPANDED_CLUSTERS


def test_graceful_underfill_after_bounded_expansion(monkeypatch):
    calls = _install_retrieval(
        monkeypatch,
        {
            0: [_candidate("p0", 0, 1.0)],
            1: [_candidate("p1", 1, 0.99)],
        },
    )

    recs = engine.recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_index(total_clusters=40),
        n=engine.TARGET_RECOMMENDATIONS,
    )

    scanned = set().union(*(set(call) for call in calls))
    assert len(recs) == 2
    assert len(scanned) <= engine.MAX_EXPANDED_CLUSTERS
    assert scanned != set(range(40))


def test_duplicate_paper_ids_are_not_selected(monkeypatch):
    candidates = _cluster_candidates(11, per_cluster=2)
    candidates[0][1] = _candidate("p0-0", 1, 0.99)
    _install_retrieval(monkeypatch, candidates, initial_clusters=list(range(11)))

    recs = engine.recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_index(total_clusters=40),
        n=engine.TARGET_RECOMMENDATIONS,
    )

    paper_ids = [rec["id"] for rec in recs]
    assert len(paper_ids) == engine.TARGET_RECOMMENDATIONS
    assert len(paper_ids) == len(set(paper_ids))
