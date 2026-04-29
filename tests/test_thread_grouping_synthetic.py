"""Fast synthetic tests for threshold-based thread grouping."""

from __future__ import annotations

import numpy as np

from user.profile import (
    SeedSignal,
    initialize_user_centroids_threshold,
    threshold_agglomerative_grouping,
)


def _unit(values: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _seed(label: str, vector: list[float]) -> SeedSignal:
    return SeedSignal(
        vector=_unit(vector),
        weight=1.0,
        reliability=1.0,
        specificity=1.0,
        split_power=1.0,
        label=label,
        source="free_text",
    )


def test_two_close_vectors_merge_at_threshold() -> None:
    seeds = [
        _seed("a", [1.0, 0.0]),
        _seed("b", [0.95, np.sqrt(1.0 - 0.95**2)]),
    ]

    groups, debug = threshold_agglomerative_grouping(
        seeds,
        merge_threshold=0.051,
        max_threads=3,
        debug=True,
    )

    assert groups == [[0, 1]]
    assert debug["phase1_threshold_merges"] == 1
    assert debug["phase2_forced_max_threads_merges"] == 0


def test_two_far_vectors_do_not_merge_above_threshold() -> None:
    seeds = [
        _seed("x", [1.0, 0.0]),
        _seed("y", [0.0, 1.0]),
    ]

    groups, debug = threshold_agglomerative_grouping(
        seeds,
        merge_threshold=0.2,
        max_threads=3,
        debug=True,
    )

    assert groups == [[0], [1]]
    assert debug["phase1_threshold_merges"] == 0
    assert debug["phase2_forced_max_threads_merges"] == 0


def test_three_vectors_two_close_one_far_produce_two_groups() -> None:
    seeds = [
        _seed("a", [1.0, 0.0]),
        _seed("b", [0.98, np.sqrt(1.0 - 0.98**2)]),
        _seed("far", [0.0, 1.0]),
    ]

    result = initialize_user_centroids_threshold(
        seeds,
        merge_threshold=0.04,
        max_threads=3,
        debug=True,
    )

    assert result.centroids.shape[0] == 2
    assert set(result.seed_labels.tolist()) == {0, 1}
    assert result.debug is not None
    assert result.debug["phase1_threshold_merges"] == 1
    assert result.debug["phase2_forced_max_threads_merges"] == 0


def test_max_threads_enforcement_only_runs_when_groups_exceed_max_threads() -> None:
    seeds = [
        _seed("e0", [1.0, 0.0, 0.0, 0.0]),
        _seed("e1", [0.0, 1.0, 0.0, 0.0]),
        _seed("e2", [0.0, 0.0, 1.0, 0.0]),
        _seed("e3", [0.0, 0.0, 0.0, 1.0]),
    ]

    _, debug_unforced = threshold_agglomerative_grouping(
        seeds,
        merge_threshold=0.01,
        max_threads=4,
        debug=True,
    )
    groups_forced, debug_forced = threshold_agglomerative_grouping(
        seeds,
        merge_threshold=0.01,
        max_threads=3,
        debug=True,
    )

    assert debug_unforced["phase1_threshold_merges"] == 0
    assert debug_unforced["phase2_forced_max_threads_merges"] == 0
    assert len(groups_forced) == 3
    assert debug_forced["phase1_threshold_merges"] == 0
    assert debug_forced["phase2_forced_max_threads_merges"] == 1


def test_initializer_output_invariants() -> None:
    seeds = [
        _seed("a", [1.0, 0.0]),
        _seed("b", [0.98, np.sqrt(1.0 - 0.98**2)]),
        _seed("far", [0.0, 1.0]),
    ]

    result = initialize_user_centroids_threshold(
        seeds,
        merge_threshold=0.04,
        max_threads=3,
    )

    assert len(result.seed_labels) == len(seeds)
    assert np.allclose(np.linalg.norm(result.centroids, axis=1), 1.0)
    assert np.isclose(result.thread_weights.sum(), 1.0)
