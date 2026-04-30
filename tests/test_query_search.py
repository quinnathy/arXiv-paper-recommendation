from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from recommender.query_search import (
    expand_query,
    lexical_score,
    search_papers,
    select_query_clusters,
    select_user_clusters,
)


def _unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _fake_index(
    embeddings: list[list[float]],
    cluster_ids: list[int],
    titles: list[str] | None = None,
) -> SimpleNamespace:
    titles = titles or [f"Paper {i}" for i in range(len(embeddings))]
    return SimpleNamespace(
        embeddings=np.stack([_unit(e) for e in embeddings]),
        cluster_ids=np.array(cluster_ids, dtype=np.int32),
        centroids=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.7, 0.7],
                [-0.7, 0.7],
                [0.7, -0.7],
                [-0.7, -0.7],
                [0.4, 0.9],
                [0.9, 0.4],
                [-0.4, 0.9],
                [0.9, -0.4],
            ],
            dtype=np.float32,
        ),
        paper_meta=[
            {
                "id": f"p{i}",
                "title": titles[i],
                "abstract": "A useful paper.",
                "categories": ["cs.LG"],
                "update_date": "2026-01-01",
                "cluster_id": cluster_ids[i],
            }
            for i in range(len(embeddings))
        ],
    )


def test_expand_query_only_expands_short_queries():
    short = expand_query("healthcare AI")
    assert short.startswith("Research papers about healthcare AI.")
    assert "benchmarks" in short

    long_query = " ".join(f"term{i}" for i in range(30))
    assert expand_query(long_query) == long_query


def test_query_and_user_cluster_budgets_are_separate():
    centroids = np.eye(12, 2, dtype=np.float32)
    query_clusters = select_query_clusters(
        np.array([1.0, 0.0], dtype=np.float32),
        centroids,
        budget=10,
    )
    user_clusters = select_user_clusters(
        np.array([[0.0, 1.0]], dtype=np.float32),
        centroids,
        diversity=1.0,
        max_budget=9,
    )

    assert len(query_clusters) == 10
    assert len(user_clusters) < len(query_clusters)


def test_search_filters_seen_ids_and_caps_same_cluster_results():
    index = _fake_index(
        embeddings=[
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.4, 0.9],
        ],
        cluster_ids=[0, 0, 0, 0, 0, 1],
    )

    results = search_papers(
        query="test",
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        user_centroids=np.array([[1.0, 0.0]], dtype=np.float32),
        index=index,
        seen_ids={"p0"},
        n=5,
        query_cluster_budget=2,
        candidate_pool_size=20,
        cluster_result_cap=3,
    )

    result_ids = [paper["id"] for paper in results]
    cluster_zero_count = sum(1 for paper in results if paper["cluster_id"] == 0)
    assert "p0" not in result_ids
    assert cluster_zero_count <= 3


def test_lexical_score_prefers_title_and_phrase_matches():
    title_match = {
        "title": "LoRA adapters for efficient fine tuning",
        "abstract": "",
    }
    abstract_match = {
        "title": "Efficient fine tuning",
        "abstract": "We study LoRA adapters for language models.",
    }
    no_match = {
        "title": "Efficient fine tuning",
        "abstract": "We study adapters for language models.",
    }

    assert lexical_score("LoRA", title_match) == 1.0
    assert lexical_score("LoRA", abstract_match) == 0.5
    assert lexical_score("LoRA", no_match) == 0.0


def test_search_includes_lexical_score_without_overpowering_query_similarity():
    index = _fake_index(
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        cluster_ids=[0, 1],
        titles=["Semantic match", "healthcare AI exact lexical match"],
    )
    index.paper_meta[0]["update_date"] = "2025-01-01"
    index.paper_meta[1]["update_date"] = "2026-01-01"

    results = search_papers(
        query="healthcare AI",
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        user_centroids=np.array([[0.0, 1.0]], dtype=np.float32),
        index=index,
        seen_ids=set(),
        n=2,
        query_cluster_budget=2,
        candidate_pool_size=2,
    )

    assert results[0]["id"] == "p0"
    assert results[1]["lexical_score"] == 1.0
    assert {
        "query_similarity",
        "user_similarity",
        "recency_score",
        "lexical_score",
    } <= set(results[0])
