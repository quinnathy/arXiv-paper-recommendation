"""Reusable Plotly utilities for embedding-space diagnostics.

Scope:
    Shared app and script helpers for turning sampled 2D embedding coordinates
    into a clean interactive dataframe and Plotly figure. This module contains
    no fitting logic and does not read or write files.

Purpose:
    Keep category parsing, top-level category mapping, hover metadata, color
    modes, and user-overlay conventions consistent between standalone PCA/UMAP
    artifacts and the optional Streamlit embedding-space panel.

Artifacts produced:
    None directly. Callers use these helpers to create:
    - data/diagnostics/pca_cluster_viz.html
    - data/diagnostics/umap_cluster_viz.html
    - Streamlit ``st.plotly_chart`` figures in the app

Command:
    This is a library module, not a CLI. Typical callers are:
    python scripts/diagnostics/visualize_clusters_pca.py --sample-size 50000
    python scripts/diagnostics/visualize_clusters_umap.py --sample-size 50000
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


UNKNOWN_CATEGORY = "unknown"
SUPPORTED_COLOR_MODES = ("primary_category", "top_level_category", "cluster")


def get_primary_category(categories) -> str:
    if categories is None:
        return UNKNOWN_CATEGORY
    if isinstance(categories, str):
        parts = categories.split()
        return parts[0] if parts else UNKNOWN_CATEGORY
    if isinstance(categories, Sequence) and not isinstance(categories, (bytes, bytearray)):
        return str(categories[0]) if categories else UNKNOWN_CATEGORY
    return str(categories)


def get_top_level_category(category: str) -> str:
    category = str(category or UNKNOWN_CATEGORY)
    if category == UNKNOWN_CATEGORY:
        return UNKNOWN_CATEGORY
    return category.split(".", 1)[0]


def _metadata_value(meta: dict, *keys: str, default: str = ""):
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return default


def build_cluster_dataframe(
    coords_2d,
    paper_indices,
    metadata,
    cluster_ids=None,
) -> pd.DataFrame:
    coords = np.asarray(coords_2d)
    indices = np.asarray(paper_indices, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_2d must have shape (N, 2), got {coords.shape}.")
    if len(indices) != len(coords):
        raise ValueError("paper_indices length must match coords_2d rows.")

    rows: list[dict] = []
    for row_i, paper_i in enumerate(indices):
        meta = metadata[int(row_i)] if len(metadata) == len(coords) else metadata[int(paper_i)]
        categories = meta.get("categories", "")
        primary = get_primary_category(categories)
        cluster_id = None
        if cluster_ids is not None:
            cluster_id = int(cluster_ids[int(paper_i)])
        elif "cluster_id" in meta:
            cluster_id = int(meta["cluster_id"])
        rows.append(
            {
                "x": float(coords[row_i, 0]),
                "y": float(coords[row_i, 1]),
                "paper_index": int(paper_i),
                "arxiv_id": _metadata_value(meta, "id", "arxiv_id"),
                "title": _metadata_value(meta, "title"),
                "primary_category": primary,
                "top_level_category": get_top_level_category(primary),
                "categories": " ".join(categories) if isinstance(categories, list) else str(categories or ""),
                "cluster": cluster_id if cluster_id is not None else -1,
                "update_date": _metadata_value(meta, "update_date", "published", "updated"),
            }
        )
    return pd.DataFrame(rows)


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for interactive embedding visualizations. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return go


def _legend_limited_series(series: pd.Series, max_legend_items: int | None) -> pd.Series:
    if max_legend_items is None or series.nunique(dropna=False) <= max_legend_items:
        return series.astype(str)
    keep = set(series.value_counts().head(max_legend_items).index.astype(str))
    return series.astype(str).where(series.astype(str).isin(keep), "Other")


def make_embedding_scatter_plot(
    df,
    color_by: str = "primary_category",
    title: str = "Paper Embedding Space",
    max_legend_items: int | None = 40,
):
    go = _require_plotly()
    if color_by not in SUPPORTED_COLOR_MODES:
        raise ValueError(f"Unsupported color_by={color_by!r}.")

    plot_df = df.copy()
    plot_df["_color"] = _legend_limited_series(plot_df[color_by], max_legend_items)
    fig = go.Figure()
    for value, group in plot_df.groupby("_color", sort=True):
        custom = np.stack(
            [
                group["arxiv_id"].astype(str),
                group["title"].astype(str),
                group["primary_category"].astype(str),
                group["categories"].astype(str),
                group["cluster"].astype(str),
                group["update_date"].astype(str),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=group["x"],
                y=group["y"],
                mode="markers",
                name=str(value),
                customdata=custom,
                marker=dict(size=4, opacity=0.55),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "arXiv: %{customdata[0]}<br>"
                    "primary: %{customdata[2]}<br>"
                    "categories: %{customdata[3]}<br>"
                    "cluster: %{customdata[4]}<br>"
                    "updated: %{customdata[5]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        legend_title_text=color_by.replace("_", " ").title(),
        margin=dict(l=32, r=24, t=58, b=32),
        hovermode="closest",
    )
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def make_user_cluster_plot(
    background_df,
    user_centroid_coords=None,
    searched_cluster_ids=None,
    served_paper_indices=None,
    current_cluster_ids=None,
    color_by: str = "primary_category",
):
    go = _require_plotly()
    fig = make_embedding_scatter_plot(
        background_df,
        color_by=color_by,
        title="Paper Embedding Space",
    )

    searched_cluster_ids = set(int(c) for c in (searched_cluster_ids or []))
    if searched_cluster_ids:
        searched = background_df[background_df["cluster"].isin(searched_cluster_ids)]
        if not searched.empty:
            fig.add_trace(
                go.Scattergl(
                    x=searched["x"],
                    y=searched["y"],
                    mode="markers",
                    name="searched clusters",
                    marker=dict(size=7, opacity=0.78, color="rgba(37, 99, 235, 0.50)"),
                    hoverinfo="skip",
                )
            )

    served_paper_indices = set(int(i) for i in (served_paper_indices or []))
    if served_paper_indices:
        served = background_df[background_df["paper_index"].isin(served_paper_indices)]
        if not served.empty:
            fig.add_trace(
                go.Scattergl(
                    x=served["x"],
                    y=served["y"],
                    mode="markers",
                    name="served papers",
                    marker=dict(
                        size=11,
                        opacity=1.0,
                        color="rgba(239, 68, 68, 0.95)",
                        line=dict(width=1.5, color="white"),
                    ),
                    text=served["title"],
                    hovertemplate="<b>%{text}</b><extra>served paper</extra>",
                )
            )

    if user_centroid_coords is not None:
        centroids = np.asarray(user_centroid_coords, dtype=np.float32)
        if centroids.ndim == 2 and centroids.shape[1] == 2 and len(centroids):
            fig.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode="markers",
                    name="user centroids",
                    marker=dict(
                        symbol="star", 
                        size=25,           
                        color="#111827", 
                        line=dict(
                            width=3,   
                            color="white"
                        )
                    ),
                    hovertemplate="user centroid<extra></extra>",
                )
            )

    if current_cluster_ids is not None:
        cluster_values = set(int(c) for c in np.asarray(current_cluster_ids).ravel())
        if cluster_values:
            current = background_df[background_df["cluster"].isin(cluster_values)]
            if not current.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=current["x"],
                        y=current["y"],
                        mode="markers",
                        name="current clusters",
                        marker=dict(size=8, opacity=0.45, color="rgba(16, 185, 129, 0.45)"),
                        hoverinfo="skip",
                    )
                )

    return fig
