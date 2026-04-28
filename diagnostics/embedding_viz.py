"""PCA and UMAP artifact generation for paper embedding visual diagnostics.

Scope:
    Shared implementation used by ``scripts/diagnostics/visualize_clusters_pca.py``,
    ``scripts/diagnostics/visualize_clusters_umap.py``, and optional offline
    pipeline visualization hooks. It loads existing embedding, metadata, and
    cluster artifacts; it never re-encodes papers.

Purpose:
    Create reusable 2D diagnostic artifacts for inspecting the corpus embedding
    space. PCA gives a quick linear sanity check. UMAP gives the preferred
    nonlinear latent-space view for the Streamlit app when ``umap-learn`` is
    installed.

Artifacts produced:
    PCA:
    - data/diagnostics/pca_cluster_viz.html
    - data/diagnostics/pca_cluster_viz.png, when static export is available
    - data/diagnostics/pca_cluster_viz_coords.npy
    - data/diagnostics/pca_cluster_viz_indices.npy
    - data/diagnostics/pca_cluster_viz_metadata.json

    UMAP:
    - data/diagnostics/umap_cluster_viz.html
    - data/diagnostics/umap_cluster_viz.png, when static export is available
    - data/diagnostics/umap_cluster_viz_coords.npy
    - data/diagnostics/umap_cluster_viz_indices.npy
    - data/diagnostics/umap_cluster_viz_model.pkl
    - data/diagnostics/umap_cluster_viz_metadata.json

Command:
    python scripts/diagnostics/visualize_clusters_pca.py --sample-size 50000
    python scripts/diagnostics/visualize_clusters_umap.py --sample-size 50000
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from diagnostics.kmeans import (
    load_embeddings,
    normalize_rows,
    sample_indices,
    utc_now_iso,
)
from recommender.visualization import build_cluster_dataframe, make_embedding_scatter_plot


def load_metadata_subset(meta_path: str | Path, paper_indices: np.ndarray) -> list[dict]:
    wanted = {int(i): pos for pos, i in enumerate(np.asarray(paper_indices, dtype=np.int64))}
    rows: list[dict | None] = [None] * len(wanted)
    found = 0
    with open(meta_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            pos = wanted.get(i)
            if pos is not None:
                rows[pos] = json.loads(line)
                found += 1
            if found == len(rows):
                break
    missing = [int(paper_indices[i]) for i, row in enumerate(rows) if row is None]
    if missing:
        raise ValueError(f"Missing metadata rows for sampled indices: {missing[:5]}")
    return [row for row in rows if row is not None]


def _load_cluster_ids(data_dir: str | Path) -> np.ndarray | None:
    path = Path(data_dir) / "cluster_ids.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_plot(fig, html_path: Path, png_path: Path | None) -> str | None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    if png_path is None:
        return None
    try:
        fig.write_image(png_path)
        return str(png_path)
    except Exception:
        warning_path = png_path.with_suffix(".png.json")
        _write_json(
            warning_path,
            {
                "warning": "PNG export requires Plotly static image support, usually `kaleido`.",
                "html_plot": str(html_path),
            },
        )
        return None


def _axis_labels(fig, explained_variance_ratio: np.ndarray | None) -> None:
    if explained_variance_ratio is None or len(explained_variance_ratio) < 2:
        fig.update_xaxes(title_text="UMAP 1")
        fig.update_yaxes(title_text="UMAP 2")
        return
    fig.update_xaxes(title_text=f"PC1 ({explained_variance_ratio[0] * 100:.2f}% variance)")
    fig.update_yaxes(title_text=f"PC2 ({explained_variance_ratio[1] * 100:.2f}% variance)")


def generate_pca_visualization(
    data_dir: str | Path = "data",
    diagnostics_dir: str | Path = "data/diagnostics",
    sample_size: int = 50_000,
    random_state: int = 42,
    color_by: str = "primary_category",
    write_png: bool = True,
) -> dict[str, Any]:
    data = Path(data_dir)
    diagnostics = Path(diagnostics_dir)
    diagnostics.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(data, mmap_mode="r")
    indices = sample_indices(int(embeddings.shape[0]), sample_size, random_state)
    x = normalize_rows(np.asarray(embeddings[indices], dtype=np.float32))

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x).astype(np.float32)
    explained = pca.explained_variance_ratio_.astype(float)

    cluster_ids = _load_cluster_ids(data)
    metadata = load_metadata_subset(data / "paper_meta.jsonl", indices)
    df = build_cluster_dataframe(coords, indices, metadata, cluster_ids=cluster_ids)
    fig = make_embedding_scatter_plot(
        df,
        color_by=color_by,
        title="PCA View of Paper Embedding Space",
    )
    _axis_labels(fig, explained)

    coords_path = diagnostics / "pca_cluster_viz_coords.npy"
    indices_path = diagnostics / "pca_cluster_viz_indices.npy"
    metadata_path = diagnostics / "pca_cluster_viz_metadata.json"
    html_path = diagnostics / "pca_cluster_viz.html"
    png_path = diagnostics / "pca_cluster_viz.png"

    np.save(coords_path, coords)
    np.save(indices_path, indices)
    exported_png = _write_plot(fig, html_path, png_path if write_png else None)

    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "method": "pca",
        "data_dir": str(data),
        "sample_size_requested": int(sample_size),
        "sample_size_used": int(len(indices)),
        "random_state": int(random_state),
        "embedding_shape": [int(v) for v in embeddings.shape],
        "explained_variance_ratio": [float(v) for v in explained],
        "color_by": color_by,
        "paths": {
            "coords": str(coords_path),
            "indices": str(indices_path),
            "metadata": str(metadata_path),
            "html": str(html_path),
            "png": exported_png or str(png_path),
        },
    }
    _write_json(metadata_path, payload)
    return payload


def generate_umap_visualization(
    data_dir: str | Path = "data",
    diagnostics_dir: str | Path = "data/diagnostics",
    sample_size: int = 50_000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    color_by: str = "primary_category",
    write_png: bool = True,
) -> dict[str, Any]:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP visualization requires `umap-learn`. Install it with "
            "`pip install umap-learn` or add it to your environment."
        ) from exc

    data = Path(data_dir)
    diagnostics = Path(diagnostics_dir)
    diagnostics.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(data, mmap_mode="r")
    indices = sample_indices(int(embeddings.shape[0]), sample_size, random_state)
    x = normalize_rows(np.asarray(embeddings[indices], dtype=np.float32))

    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = model.fit_transform(x).astype(np.float32)

    cluster_ids = _load_cluster_ids(data)
    metadata = load_metadata_subset(data / "paper_meta.jsonl", indices)
    df = build_cluster_dataframe(coords, indices, metadata, cluster_ids=cluster_ids)
    fig = make_embedding_scatter_plot(
        df,
        color_by=color_by,
        title="UMAP View of Paper Embedding Space",
    )
    _axis_labels(fig, None)

    coords_path = diagnostics / "umap_cluster_viz_coords.npy"
    indices_path = diagnostics / "umap_cluster_viz_indices.npy"
    model_path = diagnostics / "umap_cluster_viz_model.pkl"
    metadata_path = diagnostics / "umap_cluster_viz_metadata.json"
    html_path = diagnostics / "umap_cluster_viz.html"
    png_path = diagnostics / "umap_cluster_viz.png"

    np.save(coords_path, coords)
    np.save(indices_path, indices)
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    exported_png = _write_plot(fig, html_path, png_path if write_png else None)

    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "method": "umap",
        "data_dir": str(data),
        "sample_size_requested": int(sample_size),
        "sample_size_used": int(len(indices)),
        "random_state": int(random_state),
        "embedding_shape": [int(v) for v in embeddings.shape],
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "color_by": color_by,
        "paths": {
            "coords": str(coords_path),
            "indices": str(indices_path),
            "model": str(model_path),
            "metadata": str(metadata_path),
            "html": str(html_path),
            "png": exported_png or str(png_path),
        },
    }
    _write_json(metadata_path, payload)
    return payload
