"""Tests for embedding visualization utilities.

Scope:
    Fast synthetic tests for category parsing, dataframe construction, and
    Plotly figure creation when Plotly is installed.

Purpose:
    Keep the app and artifact-generation scripts aligned on primary category,
    top-level category, cluster metadata, and expected Plotly figure behavior.

Artifacts produced:
    None. These tests use tiny in-memory arrays only.

Command:
    python -m pytest tests/test_visualization_utils.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from recommender.visualization import (
    build_cluster_dataframe,
    get_primary_category,
    get_top_level_category,
    make_embedding_scatter_plot,
)


def test_primary_category_parsing():
    assert get_primary_category("cs.LG stat.ML") == "cs.LG"
    assert get_primary_category(["math.ST", "stat.TH"]) == "math.ST"
    assert get_primary_category("") == "unknown"
    assert get_primary_category(None) == "unknown"


def test_top_level_category_parsing():
    assert get_top_level_category("cs.LG") == "cs"
    assert get_top_level_category("math.ST") == "math"
    assert get_top_level_category("q-bio.QM") == "q-bio"
    assert get_top_level_category("astro-ph") == "astro-ph"


def test_build_cluster_dataframe_from_fake_metadata():
    coords = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    paper_indices = np.array([2, 5], dtype=np.int64)
    metadata = [
        {
            "id": "2401.00001",
            "title": "A paper",
            "categories": "cs.LG stat.ML",
            "update_date": "2024-01-01",
        },
        {
            "id": "2401.00002",
            "title": "Another paper",
            "categories": ["q-bio.QM"],
            "update_date": "2024-01-02",
        },
    ]
    cluster_ids = np.array([0, 0, 7, 0, 0, 9], dtype=np.int32)

    df = build_cluster_dataframe(coords, paper_indices, metadata, cluster_ids=cluster_ids)

    assert df["paper_index"].tolist() == [2, 5]
    assert df["primary_category"].tolist() == ["cs.LG", "q-bio.QM"]
    assert df["top_level_category"].tolist() == ["cs", "q-bio"]
    assert df["cluster"].tolist() == [7, 9]


def test_make_embedding_scatter_plot_returns_plotly_figure():
    if importlib.util.find_spec("plotly") is None:
        pytest.skip("Plotly is not installed in this environment.")

    coords = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    metadata = [
        {"id": "1", "title": "One", "categories": "cs.LG"},
        {"id": "2", "title": "Two", "categories": "math.ST"},
    ]
    df = build_cluster_dataframe(coords, np.array([0, 1]), metadata)
    fig = make_embedding_scatter_plot(df)

    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)
