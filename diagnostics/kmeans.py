"""Diagnostics and retraining helpers for the k-means spatial index.

Scope:
    Shared implementation used by ``scripts/diagnostics/diagnose_kmeans_k.py``,
    ``scripts/diagnostics/retrain_kmeans_index.py``, and the optional offline
    pipeline hooks. This module owns K sweep metrics, elbow estimation, cluster artifact
    validation, retraining, and production-artifact promotion.

Purpose:
    Replace the fixed ``k=500`` heuristic with a reproducible workflow for
    evaluating candidate K values and retraining the anonymous corpus-level
    MiniBatchKMeans retrieval index. The clusters are spatial buckets over
    unit-normalized SPECTER2 embeddings, not human-readable topics.

Artifacts produced:
    K sweep:
    - data/diagnostics/kmeans_k_sweep.csv
    - data/diagnostics/kmeans_k_sweep.json
    - data/diagnostics/kmeans_elbow.html
    - data/diagnostics/kmeans_elbow.png, when static export is available
    - data/diagnostics/kmeans_k_report.md

    Retraining with ``output_prefix=data/kmeans_k700``:
    - data/kmeans_k700_cluster_ids.npy
    - data/kmeans_k700_centroids.npy
    - data/kmeans_k700_cluster_sizes.npy
    - data/kmeans_k700_metadata.json

Command:
    python scripts/diagnostics/diagnose_kmeans_k.py --k-values 100 200 300 400 500 700
    python scripts/diagnostics/retrain_kmeans_index.py --k 700 --output-prefix data/kmeans_k700
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


DEFAULT_K_VALUES = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
DEFAULT_DIAGNOSTICS_DIR = Path("data/diagnostics")
PRODUCTION_CLUSTER_IDS = "cluster_ids.npy"
PRODUCTION_CENTROIDS = "centroids.npy"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_embeddings(data_dir: str | Path = "data", mmap_mode: str | None = "r") -> np.ndarray:
    path = Path(data_dir) / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {path}")
    return np.load(path, mmap_mode=mmap_mode)


def sample_indices(n: int, sample_size: int | None, random_state: int) -> np.ndarray:
    if sample_size is None or sample_size >= n:
        return np.arange(n, dtype=np.int64)
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive when provided.")
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=sample_size, replace=False).astype(np.int64))


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, eps)).astype(np.float32, copy=False)


def sampled_embeddings(
    embeddings: np.ndarray,
    sample_size: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = sample_indices(int(embeddings.shape[0]), sample_size, random_state)
    sample = np.asarray(embeddings[indices], dtype=np.float32)
    return normalize_rows(sample), indices


def estimate_elbow_k(k_values: list[int], inertia_values: list[float]) -> int | None:
    """Estimate an elbow using normalized distance from the endpoint line."""
    if len(k_values) < 3 or len(k_values) != len(inertia_values):
        return None

    points = np.column_stack([
        np.asarray(k_values, dtype=np.float64),
        np.asarray(inertia_values, dtype=np.float64),
    ])
    if not np.isfinite(points).all():
        return None

    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-12)
    norm_points = (points - mins) / spans

    start = norm_points[0]
    end = norm_points[-1]
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1e-12:
        return int(k_values[0])

    vectors = start - norm_points
    cross_z = line[0] * vectors[:, 1] - line[1] * vectors[:, 0]
    distances = np.abs(cross_z) / line_norm
    return int(k_values[int(np.argmax(distances))])


def _cluster_size_stats(labels: np.ndarray, k: int) -> dict[str, float | int]:
    sizes = np.bincount(labels.astype(np.int64), minlength=k)
    return {
        "mean_cluster_size": float(np.mean(sizes)),
        "median_cluster_size": float(np.median(sizes)),
        "min_cluster_size": int(np.min(sizes)),
        "max_cluster_size": int(np.max(sizes)),
        "empty_cluster_count": int(np.sum(sizes == 0)),
        "p05_cluster_size": float(np.percentile(sizes, 5)),
        "p95_cluster_size": float(np.percentile(sizes, 95)),
    }


def _compute_optional_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    metric_sample_size: int,
    random_state: int,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "silhouette_score_sample": None,
        "davies_bouldin_score_sample": None,
        "calinski_harabasz_score_sample": None,
    }
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return metrics

    metric_indices = sample_indices(len(x), min(metric_sample_size, len(x)), random_state)
    metric_x = np.asarray(x[metric_indices], dtype=np.float32)
    metric_labels = np.asarray(labels[metric_indices])
    if len(np.unique(metric_labels)) < 2:
        return metrics

    try:
        metrics["silhouette_score_sample"] = float(
            silhouette_score(metric_x, metric_labels, metric="cosine")
        )
    except Exception:
        metrics["silhouette_score_sample"] = None
    try:
        metrics["davies_bouldin_score_sample"] = float(
            davies_bouldin_score(metric_x, metric_labels)
        )
    except Exception:
        metrics["davies_bouldin_score_sample"] = None
    try:
        metrics["calinski_harabasz_score_sample"] = float(
            calinski_harabasz_score(metric_x, metric_labels)
        )
    except Exception:
        metrics["calinski_harabasz_score_sample"] = None
    return metrics


def detect_current_production_k(data_dir: str | Path = "data") -> int | None:
    centroids_path = Path(data_dir) / PRODUCTION_CENTROIDS
    if not centroids_path.exists():
        return None
    centroids = np.load(centroids_path, mmap_mode="r")
    if centroids.ndim != 2:
        return None
    return int(centroids.shape[0])


def run_k_sweep(
    data_dir: str | Path = "data",
    diagnostics_dir: str | Path = DEFAULT_DIAGNOSTICS_DIR,
    k_values: list[int] | None = None,
    sample_size: int | None = 200_000,
    batch_size: int = 8192,
    max_iter: int = 100,
    random_state: int = 42,
    metric_sample_size: int = 10_000,
    compute_quality_metrics: bool = True,
) -> dict[str, Any]:
    """Fit MiniBatchKMeans for many K values and persist diagnostic outputs."""
    k_values = k_values or DEFAULT_K_VALUES
    embeddings = load_embeddings(data_dir, mmap_mode="r")
    x, indices = sampled_embeddings(embeddings, sample_size, random_state)

    results: list[dict[str, Any]] = []
    for k in k_values:
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}.")
        if k > len(x):
            raise ValueError(f"K={k} exceeds sampled rows={len(x)}.")

        model = MiniBatchKMeans(
            n_clusters=int(k),
            n_init=5,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        start = time.perf_counter()
        model.fit(x)
        fit_seconds = time.perf_counter() - start
        labels = model.labels_.astype(np.int32, copy=False)

        row: dict[str, Any] = {
            "k": int(k),
            "inertia": float(model.inertia_),
            "inertia_per_sample": float(model.inertia_ / len(x)),
            "fit_seconds": float(fit_seconds),
        }
        row.update(_cluster_size_stats(labels, int(k)))
        if compute_quality_metrics:
            row.update(
                _compute_optional_metrics(
                    x=x,
                    labels=labels,
                    metric_sample_size=metric_sample_size,
                    random_state=random_state,
                )
            )
        results.append(row)

    estimated_elbow = estimate_elbow_k(
        [int(r["k"]) for r in results],
        [float(r["inertia"]) for r in results],
    )
    current_k = detect_current_production_k(data_dir)

    diagnostics = Path(diagnostics_dir)
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "kmeans_k_sweep.csv"
    json_path = diagnostics / "kmeans_k_sweep.json"
    html_path = diagnostics / "kmeans_elbow.html"
    png_path = diagnostics / "kmeans_elbow.png"
    report_path = diagnostics / "kmeans_k_report.md"

    _write_sweep_csv(csv_path, results)
    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "data_dir": str(data_dir),
        "sample_size_requested": sample_size,
        "sample_size_used": int(len(x)),
        "sample_indices_path": None,
        "embedding_shape": [int(v) for v in embeddings.shape],
        "k_values": [int(k) for k in k_values],
        "estimated_elbow_k": estimated_elbow,
        "current_production_k": current_k,
        "random_state": int(random_state),
        "batch_size": int(batch_size),
        "max_iter": int(max_iter),
        "metric_sample_size": int(metric_sample_size),
        "sklearn_version": sklearn_version,
        "results": results,
        "paths": {
            "csv": str(csv_path),
            "json": str(json_path),
            "html": str(html_path),
            "png": str(png_path),
            "report": str(report_path),
        },
    }
    if len(indices) != int(embeddings.shape[0]):
        sample_indices_path = diagnostics / "kmeans_k_sweep_sample_indices.npy"
        np.save(sample_indices_path, indices)
        payload["sample_indices_path"] = str(sample_indices_path)

    _write_json(json_path, payload)
    make_elbow_plots(results, html_path, png_path, current_k=current_k, estimated_elbow_k=estimated_elbow)
    write_kmeans_report(report_path, payload)
    _write_json(json_path, payload)
    return payload


def _write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _relative_improvements(results: list[dict[str, Any]]) -> list[float | None]:
    improvements: list[float | None] = [None]
    for prev, cur in zip(results, results[1:]):
        prev_inertia = float(prev["inertia"])
        cur_inertia = float(cur["inertia"])
        if prev_inertia <= 0:
            improvements.append(None)
        else:
            improvements.append(float((prev_inertia - cur_inertia) / prev_inertia))
    return improvements


def make_elbow_plots(
    results: list[dict[str, Any]],
    html_path: str | Path,
    png_path: str | Path,
    current_k: int | None = None,
    estimated_elbow_k: int | None = None,
) -> None:
    if not results:
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for k-means elbow plots. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    k = [int(r["k"]) for r in results]
    rel = _relative_improvements(results)
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "K vs inertia",
            "K vs inertia per sample",
            "Relative improvement from previous K",
            "K vs fit time",
            "K vs silhouette score",
            "K vs cluster size",
        ),
    )
    fig.add_trace(go.Scatter(x=k, y=[r["inertia"] for r in results], mode="lines+markers", name="inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=[r["inertia_per_sample"] for r in results], mode="lines+markers", name="inertia/sample"), row=1, col=2)
    fig.add_trace(go.Scatter(x=k, y=rel, mode="lines+markers", name="relative improvement"), row=2, col=1)
    fig.add_trace(go.Scatter(x=k, y=[r["fit_seconds"] for r in results], mode="lines+markers", name="fit seconds"), row=2, col=2)
    fig.add_trace(go.Scatter(x=k, y=[r.get("silhouette_score_sample") for r in results], mode="lines+markers", name="silhouette"), row=3, col=1)
    fig.add_trace(go.Scatter(x=k, y=[r["median_cluster_size"] for r in results], mode="lines+markers", name="median cluster size"), row=3, col=2)
    fig.add_trace(go.Scatter(x=k, y=[r["p95_cluster_size"] for r in results], mode="lines+markers", name="p95 cluster size"), row=3, col=2)

    for marker_k, label, color in (
        (current_k, "current production K", "#ef4444"),
        (estimated_elbow_k, "estimated elbow K", "#2563eb"),
    ):
        if marker_k is None:
            continue
        for row in range(1, 4):
            for col in range(1, 3):
                fig.add_vline(
                    x=marker_k,
                    line_width=1,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=label if row == 1 and col == 1 else None,
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title="MiniBatchKMeans K Sweep Diagnostics",
        template="plotly_white",
        height=950,
        legend_title_text="Metric",
        margin=dict(l=48, r=32, t=86, b=48),
    )
    fig.update_xaxes(title_text="K")
    html_path = Path(html_path)
    png_path = Path(png_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(png_path)
    except Exception:
        payload = {
            "warning": "PNG export requires Plotly static image support, usually `kaleido`.",
            "html_plot": str(html_path),
        }
        with open(png_path.with_suffix(".png.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def write_kmeans_report(path: str | Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    recommended = payload.get("estimated_elbow_k")
    current_k = payload.get("current_production_k")

    inertia_rows = [
        "| K | Inertia | Inertia/sample | Relative improvement | Fit seconds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row, rel in zip(results, _relative_improvements(results)):
        inertia_rows.append(
            f"| {row['k']} | {row['inertia']:.3f} | {row['inertia_per_sample']:.6f} | "
            f"{'' if rel is None else f'{rel:.4f}'} | {row['fit_seconds']:.2f} |"
        )

    size_rows = [
        "| K | Mean | Median | Min | P05 | P95 | Max | Empty |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        size_rows.append(
            f"| {row['k']} | {row['mean_cluster_size']:.1f} | {row['median_cluster_size']:.1f} | "
            f"{row['min_cluster_size']} | {row['p05_cluster_size']:.1f} | "
            f"{row['p95_cluster_size']:.1f} | {row['max_cluster_size']} | "
            f"{row['empty_cluster_count']} |"
        )

    paths = payload.get("paths", {})
    text = "\n".join(
        [
            "# K-means K Diagnostic Report",
            "",
            f"Generated: {payload.get('created_at')}",
            "",
            f"K grid tested: {payload.get('k_values')}",
            f"Estimated elbow K: {recommended}",
            f"Current production K: {current_k}",
            f"Recommended diagnostic candidate: {recommended}",
            "",
            "This recommendation is diagnostic, not an objective optimum. The k-means clusters are anonymous spatial-index buckets used to reduce retrieval work before exact dot-product scoring.",
            "",
            "## Inertia",
            "",
            *inertia_rows,
            "",
            "## Cluster Size Distribution",
            "",
            *size_rows,
            "",
            "## Retrieval Tradeoff Notes",
            "",
            "- Smaller K means larger clusters, slower per-query exact scoring, and potentially more robust neighborhoods.",
            "- Larger K means smaller clusters and faster exact scoring, but can fragment neighborhoods and may require searching more clusters for similar recall.",
            "- The chosen K should balance retrieval efficiency, compactness, cluster size distribution, stability, and compatibility with current artifacts.",
            "",
            "## Generated Plots",
            "",
            f"- HTML elbow plot: {paths.get('html')}",
            f"- PNG elbow plot: {paths.get('png')}",
            f"- CSV metrics: {paths.get('csv')}",
            f"- JSON metrics: {paths.get('json')}",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_cluster_artifacts(
    cluster_ids: np.ndarray,
    centroids: np.ndarray,
    cluster_sizes: np.ndarray,
    n_rows: int,
    k: int,
    *,
    norm_atol: float = 1e-4,
) -> None:
    if cluster_ids.shape != (n_rows,):
        raise ValueError(f"cluster_ids.shape={cluster_ids.shape} != ({n_rows},)")
    if centroids.shape[0] != k or centroids.ndim != 2:
        raise ValueError(f"centroids.shape={centroids.shape} is incompatible with K={k}.")
    if cluster_sizes.shape != (k,):
        raise ValueError(f"cluster_sizes.shape={cluster_sizes.shape} != ({k},)")
    if cluster_ids.size and (cluster_ids.min() < 0 or cluster_ids.max() >= k):
        raise ValueError("cluster_ids must all be in [0, K - 1].")
    if not np.isfinite(centroids).all():
        raise ValueError("Centroids contain non-finite values.")
    norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(norms, 1.0, atol=norm_atol):
        raise ValueError("Centroid row norms are not close to 1.")
    if int(cluster_sizes.sum()) != int(n_rows):
        raise ValueError(f"Cluster size sum {cluster_sizes.sum()} != N={n_rows}.")


def retrain_kmeans_index(
    k: int,
    data_dir: str | Path = "data",
    output_prefix: str | Path | None = None,
    batch_size: int = 8192,
    max_iter: int = 100,
    random_state: int = 42,
    make_current: bool = False,
) -> dict[str, Any]:
    embeddings = load_embeddings(data_dir, mmap_mode="r")
    if k <= 0 or k > int(embeddings.shape[0]):
        raise ValueError(f"K must be in [1, N], got K={k}, N={embeddings.shape[0]}.")

    model = MiniBatchKMeans(
        n_clusters=int(k),
        n_init=5,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
    )
    start = time.perf_counter()
    model.fit(embeddings)
    fit_seconds = time.perf_counter() - start

    cluster_ids = model.labels_.astype(np.int32)
    centroids = normalize_rows(model.cluster_centers_.astype(np.float32))
    cluster_sizes = np.bincount(cluster_ids, minlength=k).astype(np.int64)
    validate_cluster_artifacts(cluster_ids, centroids, cluster_sizes, int(embeddings.shape[0]), k)

    prefix = Path(output_prefix) if output_prefix else Path(data_dir) / f"kmeans_k{k}"
    paths = {
        "cluster_ids": prefix.with_name(prefix.name + "_cluster_ids.npy"),
        "centroids": prefix.with_name(prefix.name + "_centroids.npy"),
        "cluster_sizes": prefix.with_name(prefix.name + "_cluster_sizes.npy"),
        "metadata": prefix.with_name(prefix.name + "_metadata.json"),
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    np.save(paths["cluster_ids"], cluster_ids)
    np.save(paths["centroids"], centroids)
    np.save(paths["cluster_sizes"], cluster_sizes)
    metadata: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "k": int(k),
        "random_state": int(random_state),
        "batch_size": int(batch_size),
        "max_iter": int(max_iter),
        "fit_seconds": float(fit_seconds),
        "inertia": float(model.inertia_),
        "sklearn_version": sklearn_version,
        "data_shape": [int(v) for v in embeddings.shape],
        "centroid_norm_min": float(np.linalg.norm(centroids, axis=1).min()),
        "centroid_norm_max": float(np.linalg.norm(centroids, axis=1).max()),
        "paths": {key: str(value) for key, value in paths.items()},
        "made_current": bool(make_current),
    }
    if make_current:
        metadata["current_paths"] = make_kmeans_artifacts_current(
            data_dir=data_dir,
            cluster_ids_path=paths["cluster_ids"],
            centroids_path=paths["centroids"],
            cluster_ids=cluster_ids,
        )
    _write_json(paths["metadata"], metadata)
    return metadata


def make_kmeans_artifacts_current(
    data_dir: str | Path,
    cluster_ids_path: str | Path,
    centroids_path: str | Path,
    cluster_ids: np.ndarray | None = None,
) -> dict[str, str]:
    data = Path(data_dir)
    current_cluster_ids = data / PRODUCTION_CLUSTER_IDS
    current_centroids = data / PRODUCTION_CENTROIDS
    shutil.copy2(cluster_ids_path, current_cluster_ids)
    shutil.copy2(centroids_path, current_centroids)

    paths = {
        "cluster_ids": str(current_cluster_ids),
        "centroids": str(current_centroids),
    }
    meta_path = data / "paper_meta.jsonl"
    if cluster_ids is not None and meta_path.exists():
        backup_path = data / "paper_meta_before_kmeans_make_current.jsonl"
        if not backup_path.exists():
            shutil.copy2(meta_path, backup_path)
        update_paper_metadata_cluster_ids(meta_path, cluster_ids)
        paths["paper_meta_backup"] = str(backup_path)
        paths["paper_meta"] = str(meta_path)
    return paths


def update_paper_metadata_cluster_ids(meta_path: str | Path, cluster_ids: np.ndarray) -> None:
    path = Path(meta_path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    row_count = 0
    with open(path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for row_count, line in enumerate(src, start=1):
            if row_count > len(cluster_ids):
                tmp.unlink(missing_ok=True)
                raise ValueError(f"Metadata rows exceed cluster ids length={len(cluster_ids)}.")
            paper = json.loads(line)
            paper["cluster_id"] = int(cluster_ids[row_count - 1])
            dst.write(json.dumps(paper) + "\n")
    if row_count != len(cluster_ids):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Metadata rows {row_count} != cluster ids {len(cluster_ids)}.")
    tmp.replace(path)
