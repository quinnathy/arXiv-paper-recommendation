"""Barebones Workspace paper-position visualization."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from pipeline.index import PaperIndex
from recommender.visualization import get_primary_category


VIZ_ARTIFACTS = (
    {
        "name": "PCA",
        "coords": Path("data/diagnostics/pca_cluster_viz_coords.npy"),
        "indices": Path("data/diagnostics/pca_cluster_viz_indices.npy"),
    },
)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _paper_index_by_id(index: PaperIndex, arxiv_id: str) -> int | None:
    for i, meta in enumerate(index.paper_meta):
        if meta.get("id") == arxiv_id:
            return i
    return None


def _short_title(title: str, max_words: int = 2) -> str:
    words = " ".join(str(title or "").split()).split()
    if not words:
        return "Paper"
    return " ".join(words[:max_words])


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = float(np.abs(coords).max())
    if scale > 1e-12:
        coords = coords / scale
    return coords


def _fallback_paper_coords(paper_vectors: np.ndarray) -> tuple[np.ndarray, str]:
    if len(paper_vectors) == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32), "fallback"
    if len(paper_vectors) == 2:
        return np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32), "fallback"

    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=42).fit_transform(paper_vectors)
    return coords.astype(np.float32), "fallback PCA"


def _workspace_pca_coords(
    index: PaperIndex,
    paper_indices: list[int],
    paper_vectors: np.ndarray,
) -> tuple[np.ndarray, str]:
    for artifact in VIZ_ARTIFACTS:
        if not artifact["coords"].exists() or not artifact["indices"].exists():
            continue

        coords = np.load(artifact["coords"])
        sampled_indices = np.load(artifact["indices"]).astype(np.int64)
        if len(coords) != len(sampled_indices):
            continue
        if len(sampled_indices) == 0:
            continue
        if int(sampled_indices.max()) >= len(index.embeddings):
            continue

        sampled_pos = {int(paper_idx): pos for pos, paper_idx in enumerate(sampled_indices)}
        result: list[np.ndarray | None] = []
        missing: list[tuple[int, np.ndarray]] = []
        for pos, paper_idx in enumerate(paper_indices):
            coord_pos = sampled_pos.get(int(paper_idx))
            if coord_pos is None:
                result.append(None)
                missing.append((pos, paper_vectors[pos]))
            else:
                result.append(np.asarray(coords[coord_pos], dtype=np.float32))

        if missing:
            sample_vectors = _unit_rows(
                np.asarray(index.embeddings[sampled_indices], dtype=np.float32)
            )
            for pos, vector in missing:
                sims = sample_vectors @ vector
                k = min(25, len(sample_vectors))
                top = np.argpartition(sims, -k)[-k:]
                top_sims = sims[top]
                weights = np.exp((top_sims - top_sims.max()) * 20.0)
                weights = weights / np.maximum(weights.sum(), 1e-12)
                result[pos] = (coords[top] * weights[:, None]).sum(axis=0).astype(np.float32)

        if all(coord is not None for coord in result):
            return _normalize_coords(np.stack(result)), str(artifact["name"])

    coords, source = _fallback_paper_coords(paper_vectors)
    return _normalize_coords(coords), source


def build_workspace_concept_map(
    index: PaperIndex,
    workspace_ids: list[str],
    *_args,
    **_kwargs,
) -> dict:
    """Build a minimal paper-position map for the current Workspace."""
    paper_indices = [
        idx
        for arxiv_id in workspace_ids
        if (idx := _paper_index_by_id(index, arxiv_id)) is not None
    ]
    if not paper_indices:
        return {
            "nodes": [],
            "edges": [],
            "summary": {"position_source": None},
            "counts": {"papers": 0, "concepts": 0},
        }

    paper_vectors = _unit_rows(np.asarray(index.embeddings[paper_indices], dtype=np.float32))
    paper_coords, position_source = _workspace_pca_coords(index, paper_indices, paper_vectors)
    paper_metas = [index.paper_meta[paper_idx] for paper_idx in paper_indices]

    nodes: list[dict] = []
    for pos, meta in enumerate(paper_metas):
        arxiv_id = str(meta.get("id", workspace_ids[pos]))
        title = meta.get("title", arxiv_id)
        nodes.append(
            {
                "id": f"paper:{arxiv_id}",
                "key": arxiv_id,
                "label": _short_title(title),
                "title": title,
                "type": "paper",
                "primary_category": get_primary_category(meta.get("categories", "")),
                "x": float(paper_coords[pos, 0]),
                "y": float(paper_coords[pos, 1]),
            }
        )

    return {
        "nodes": nodes,
        "edges": [],
        "summary": {
            "position_source": position_source,
        },
        "counts": {
            "papers": len(nodes),
            "concepts": 0,
        },
    }


def make_workspace_concept_map_figure(graph: dict):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for Workspace Visualization. Install dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    nodes = [node for node in graph.get("nodes", []) if node.get("type") == "paper"]
    fig = go.Figure()

    if nodes:
        fig.add_trace(
            go.Scatter(
                x=[node["x"] for node in nodes],
                y=[node["y"] for node in nodes],
                mode="markers+text",
                name="Paper",
                text=[node["label"] for node in nodes],
                textposition="top center",
                marker=dict(
                    symbol="circle",
                    size=20,
                    color="#2563eb",
                    line=dict(width=1.5, color="white"),
                ),
                customdata=[
                    [
                        node.get("title", node["label"]),
                        node.get("key", ""),
                        node.get("primary_category", "unknown"),
                    ]
                    for node in nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "arXiv: %{customdata[1]}<br>"
                    "category: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Workspace Papers", y=0.98, yanchor="top"),
        template="plotly_white",
        height=760,
        margin=dict(l=18, r=18, t=72, b=18),
        hovermode="closest",
        showlegend=False,
    )
    fig.update_xaxes(visible=False, zeroline=False)
    fig.update_yaxes(visible=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig
