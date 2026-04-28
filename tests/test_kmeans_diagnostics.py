"""Tests for k-means diagnostic helpers.

Scope:
    Fast synthetic tests for elbow estimation and cluster artifact validation.

Purpose:
    Guard the small reusable pieces that decide/report K candidates and ensure
    retrained cluster artifacts remain compatible with recommender assumptions.

Artifacts produced:
    None. These tests use tiny in-memory arrays only.

Command:
    python -m pytest tests/test_kmeans_diagnostics.py
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.kmeans import estimate_elbow_k, validate_cluster_artifacts


def test_estimate_elbow_k_on_synthetic_curve():
    k_values = [100, 200, 300, 400, 500, 700, 1000]
    inertia = [1000.0, 650.0, 470.0, 390.0, 350.0, 330.0, 318.0]

    assert estimate_elbow_k(k_values, inertia) in {300, 400}


def test_validate_cluster_artifacts_passes_on_well_formed_arrays():
    cluster_ids = np.array([0, 1, 1, 2], dtype=np.int32)
    centroids = np.eye(3, dtype=np.float32)
    cluster_sizes = np.array([1, 2, 1], dtype=np.int64)

    validate_cluster_artifacts(cluster_ids, centroids, cluster_sizes, n_rows=4, k=3)


def test_validate_cluster_artifacts_catches_shape_mismatch():
    cluster_ids = np.array([0, 1], dtype=np.int32)
    centroids = np.eye(3, dtype=np.float32)
    cluster_sizes = np.array([1, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="cluster_ids.shape"):
        validate_cluster_artifacts(cluster_ids, centroids, cluster_sizes, n_rows=3, k=3)


def test_validate_cluster_artifacts_catches_bad_centroid_norms():
    cluster_ids = np.array([0, 1], dtype=np.int32)
    centroids = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cluster_sizes = np.array([1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="norms"):
        validate_cluster_artifacts(cluster_ids, centroids, cluster_sizes, n_rows=2, k=2)
