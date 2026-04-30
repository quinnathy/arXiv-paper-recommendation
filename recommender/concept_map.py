"""Workspace concept-map graph generation and rendering."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from pipeline.concept_tags import CONCEPT_TAG_MAP
from pipeline.index import PaperIndex
from recommender.visualization import get_primary_category


KEYWORD_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "based",
    "been",
    "being",
    "between",
    "both",
    "from",
    "have",
    "into",
    "more",
    "most",
    "over",
    "paper",
    "papers",
    "present",
    "propose",
    "proposed",
    "result",
    "results",
    "show",
    "shown",
    "such",
    "than",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
}

VIZ_ARTIFACTS = (
    {
        "name": "PCA",
        "coords": Path("data/diagnostics/pca_cluster_viz_coords.npy"),
        "indices": Path("data/diagnostics/pca_cluster_viz_indices.npy"),
    },
)
CONCEPT_RETENTION_THRESHOLD = 0.5


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


def _concept_label(concept_key: str) -> str:
    concept = CONCEPT_TAG_MAP.get(concept_key)
    if concept is None:
        return concept_key.replace("_", " ").title()
    return concept.label


def _paper_categories(meta: dict) -> set[str]:
    categories = meta.get("categories", [])
    if isinstance(categories, str):
        return {category for category in categories.split() if category}
    return {str(category) for category in categories if category}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _keyword_tokens(meta: dict) -> set[str]:
    text = f"{meta.get('title', '')} {meta.get('abstract', '')}".lower()
    normalized = "".join(char if char.isalnum() else " " for char in text)
    return {
        token
        for token in normalized.split()
        if len(token) >= 4 and token not in KEYWORD_STOPWORDS and not token.isdigit()
    }


def _top_concept_keys(
    concept_sims: np.ndarray | None,
    concept_keys: list[str],
    paper_pos: int,
    top_k: int = 4,
    threshold: float = 0.3,
) -> set[str]:
    if concept_sims is None or not concept_keys:
        return set()
    scores = concept_sims[paper_pos]
    top = np.argsort(scores)[::-1][:top_k]
    return {
        concept_keys[int(pos)]
        for pos in top
        if float(scores[int(pos)]) >= threshold
    }


def _paper_connection_score(
    left_meta: dict,
    right_meta: dict,
    embedding_similarity: float,
    left_concepts: set[str],
    right_concepts: set[str],
) -> dict:
    left_categories = _paper_categories(left_meta)
    right_categories = _paper_categories(right_meta)
    left_keywords = _keyword_tokens(left_meta)
    right_keywords = _keyword_tokens(right_meta)

    category_overlap = _jaccard(left_categories, right_categories)
    concept_overlap = _jaccard(left_concepts, right_concepts)
    keyword_overlap = _jaccard(left_keywords, right_keywords)
    score = (
        0.70 * embedding_similarity
        + 0.20 * category_overlap
        + 0.10 * keyword_overlap
    )
    if left_concepts or right_concepts:
        score = (
            0.55 * embedding_similarity
            + 0.20 * category_overlap
            + 0.15 * concept_overlap
            + 0.10 * keyword_overlap
        )

    shared_categories = sorted(left_categories & right_categories)
    shared_concepts = sorted(_concept_label(key) for key in left_concepts & right_concepts)
    shared_keywords = sorted(left_keywords & right_keywords)[:8]
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "embedding_similarity": float(embedding_similarity),
        "category_overlap": float(category_overlap),
        "concept_overlap": float(concept_overlap),
        "keyword_overlap": float(keyword_overlap),
        "shared_categories": shared_categories,
        "shared_concepts": shared_concepts,
        "shared_keywords": shared_keywords,
    }


def _fallback_paper_coords(paper_vectors: np.ndarray) -> tuple[np.ndarray, str]:
    if len(paper_vectors) == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32), "fallback"
    if len(paper_vectors) == 2:
        return np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32), "fallback"

    from sklearn.decomposition import PCA

    return (
        PCA(n_components=2, random_state=42).fit_transform(paper_vectors).astype(np.float32),
        "fallback PCA",
    )


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = float(np.abs(coords).max())
    if scale > 1e-12:
        coords = coords / scale
    return coords


def _spread_overlapping_coords(
    coords: np.ndarray,
    min_distance: float = 0.34,
    iterations: int = 80,
) -> np.ndarray:
    coords = _normalize_coords(coords)
    if len(coords) < 2:
        return coords

    for _ in range(iterations):
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        moved = False
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                distance = float(distances[i, j])
                if distance >= min_distance:
                    continue
                if distance < 1e-9:
                    direction = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    direction = delta[i, j] / distance
                shift = direction * ((min_distance - distance) / 2.0)
                coords[i] += shift
                coords[j] -= shift
                moved = True
        if not moved:
            break

    return _normalize_coords(coords)


def _custom_distance_coords(
    custom_paper_sims: np.ndarray,
    initial_coords: np.ndarray,
) -> tuple[np.ndarray, str]:
    if len(custom_paper_sims) <= 2:
        return _spread_overlapping_coords(initial_coords), "custom distance layout"

    distances = 1.0 - np.clip(custom_paper_sims, 0.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    try:
        from sklearn.manifold import MDS

        initial = _normalize_coords(initial_coords)
        try:
            coords = MDS(
                n_components=2,
                metric_mds=True,
                metric="precomputed",
                init="random",
                random_state=42,
                max_iter=300,
                eps=1e-4,
                n_init=1,
            ).fit_transform(distances, init=initial)
        except TypeError:
            coords = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                max_iter=300,
                eps=1e-4,
                n_init=1,
            ).fit_transform(distances, init=initial)
        return _spread_overlapping_coords(coords), "custom distance MDS"
    except Exception:
        return _spread_overlapping_coords(initial_coords), "PCA fallback"


def _embedding_space_coords(
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
        if len(sampled_indices) and int(sampled_indices.max()) >= len(index.embeddings):
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


def _theme_components(paper_sims: np.ndarray, threshold: float) -> list[int]:
    n = int(paper_sims.shape[0])
    labels = [-1] * n
    current = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current
        while stack:
            i = stack.pop()
            neighbors = np.where(paper_sims[i] >= threshold)[0]
            for j in neighbors:
                j = int(j)
                if i == j or labels[j] != -1:
                    continue
                labels[j] = current
                stack.append(j)
        current += 1
    return labels


def _paper_summary(nodes: list[dict], paper_sims: np.ndarray) -> dict:
    paper_nodes = [node for node in nodes if node["type"] == "paper"]
    if not paper_nodes:
        return {}
    if len(paper_nodes) == 1:
        return {
            "coherence": "single paper",
            "average_similarity": None,
            "closest_pair": None,
            "most_isolated": paper_nodes[0]["title"],
        }

    upper = paper_sims[np.triu_indices(len(paper_nodes), k=1)]
    avg = float(np.mean(upper)) if len(upper) else 0.0
    coherence = "high" if avg >= 0.72 else "medium" if avg >= 0.5 else "mixed"

    masked = paper_sims.copy()
    np.fill_diagonal(masked, -np.inf)
    closest_i, closest_j = np.unravel_index(np.argmax(masked), masked.shape)
    nearest_scores = masked.max(axis=1)
    isolated_i = int(np.argmin(nearest_scores))

    return {
        "coherence": coherence,
        "average_similarity": avg,
        "closest_pair": {
            "source": paper_nodes[int(closest_i)]["title"],
            "target": paper_nodes[int(closest_j)]["title"],
            "similarity": float(masked[closest_i, closest_j]),
        },
        "most_isolated": paper_nodes[isolated_i]["title"],
    }


def _build_concept_anchors(
    concept_embeddings: dict[str, np.ndarray] | None,
    paper_vectors: np.ndarray,
    concept_top_k: int,
    concept_similarity_threshold: float,
    max_anchor_nodes: int,
    retention_threshold: float = CONCEPT_RETENTION_THRESHOLD,
) -> tuple[list[dict], np.ndarray | None, list[str]]:
    if not concept_embeddings:
        return [], None, []

    keys = list(concept_embeddings.keys())
    matrix = _unit_rows(np.stack([concept_embeddings[key] for key in keys]).astype(np.float32))
    sims = paper_vectors @ matrix.T
    candidates: dict[str, dict] = {}
    retention_counts = {
        key: int(np.sum(sims[:, key_pos] >= retention_threshold))
        for key_pos, key in enumerate(keys)
    }

    for paper_pos in range(sims.shape[0]):
        top = np.argsort(sims[paper_pos])[::-1][:concept_top_k]
        for anchor_pos in top:
            score = float(sims[paper_pos, anchor_pos])
            if score < retention_threshold:
                continue
            key = keys[int(anchor_pos)]
            if retention_counts[key] < 2:
                continue
            node_id = f"concept:{key}"
            current = candidates.get(node_id)
            if current is None or score > current["score"]:
                candidates[node_id] = {
                    "id": node_id,
                    "key": key,
                    "label": _concept_label(key),
                    "title": _concept_label(key),
                    "type": "concept",
                    "vector": matrix[int(anchor_pos)],
                    "score": score,
                    "retention_count": retention_counts[key],
                    "matrix_pos": int(anchor_pos),
                }

    for anchor_pos, key in enumerate(keys):
        if retention_counts[key] < 2:
            continue
        node_id = f"concept:{key}"
        if node_id in candidates:
            continue
        score = float(np.max(sims[:, anchor_pos]))
        candidates[node_id] = {
            "id": node_id,
            "key": key,
            "label": _concept_label(key),
            "title": _concept_label(key),
            "type": "concept",
            "vector": matrix[int(anchor_pos)],
            "score": score,
            "retention_count": retention_counts[key],
            "matrix_pos": int(anchor_pos),
        }

    anchors = sorted(
        candidates.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:max_anchor_nodes]
    return anchors, sims, keys


def build_workspace_concept_map(
    index: PaperIndex,
    workspace_ids: list[str],
    paper_top_k: int = 3,
    paper_similarity_threshold: float = 0.5,
    concept_top_k: int = 3,
    concept_similarity_threshold: float = 0.5,
    max_anchor_nodes: int = 12,
    theme_similarity_threshold: float = 0.55,
) -> dict:
    """Build a small concept map for the current Workspace."""
    paper_indices = [
        idx
        for arxiv_id in workspace_ids
        if (idx := _paper_index_by_id(index, arxiv_id)) is not None
    ]
    if not paper_indices:
        return {"nodes": [], "edges": []}

    paper_vectors = _unit_rows(np.asarray(index.embeddings[paper_indices], dtype=np.float32))
    paper_coords, position_source = _embedding_space_coords(index, paper_indices, paper_vectors)
    paper_sims = paper_vectors @ paper_vectors.T
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
                "vector": paper_vectors[pos],
                "x": float(paper_coords[pos, 0]),
                "y": float(paper_coords[pos, 1]),
            }
        )

    edges: list[dict] = []
    anchors, concept_sims, concept_keys = _build_concept_anchors(
        index.concept_embeddings,
        paper_vectors,
        concept_top_k,
        concept_similarity_threshold,
        max_anchor_nodes,
    )

    paper_concept_sets = [
        _top_concept_keys(
            concept_sims,
            concept_keys,
            paper_pos,
            top_k=concept_top_k,
            threshold=concept_similarity_threshold,
        )
        for paper_pos in range(len(paper_vectors))
    ]
    custom_paper_sims = np.eye(len(paper_vectors), dtype=np.float32)
    if len(paper_vectors) > 1:
        pair_scores: dict[tuple[int, int], dict] = {}
        for i in range(len(paper_vectors)):
            for j in range(i + 1, len(paper_vectors)):
                score_parts = _paper_connection_score(
                    paper_metas[i],
                    paper_metas[j],
                    float(paper_sims[i, j]),
                    paper_concept_sets[i],
                    paper_concept_sets[j],
                )
                pair_scores[(i, j)] = score_parts
                custom_paper_sims[i, j] = score_parts["score"]
                custom_paper_sims[j, i] = score_parts["score"]

        seen_pairs: set[tuple[str, str]] = set()
        for i, node in enumerate(nodes):
            neighbors = np.argsort(custom_paper_sims[i])[::-1]
            kept = 0
            for j in neighbors:
                j = int(j)
                if i == j or float(custom_paper_sims[i, j]) < paper_similarity_threshold:
                    continue
                pair = tuple(sorted((node["id"], nodes[j]["id"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                score_parts = pair_scores[tuple(sorted((i, j)))]
                edges.append(
                    {
                        "source": pair[0],
                        "target": pair[1],
                        "weight": score_parts["score"],
                        "type": "paper_similarity",
                        **score_parts,
                    }
                )
                kept += 1
                if kept >= paper_top_k:
                    break

    theme_labels = _theme_components(custom_paper_sims, theme_similarity_threshold)
    paper_coords, layout_source = _custom_distance_coords(custom_paper_sims, paper_coords)
    position_source = f"{layout_source} using {position_source} positions"
    for pos, node in enumerate(nodes):
        node["theme"] = int(theme_labels[pos])
        node["x"] = float(paper_coords[pos, 0])
        node["y"] = float(paper_coords[pos, 1])

    strongest_anchor_by_theme: dict[int, tuple[float, str]] = {}
    if concept_sims is not None:
        edge_keys: set[tuple[str, str]] = set()

        def add_concept_edge(
            paper_pos: int,
            paper_node: dict,
            anchor: dict,
            score: float,
            retained_by_floor: bool = False,
        ) -> None:
            edge_key = (paper_node["id"], anchor["id"])
            if edge_key in edge_keys:
                return
            edge_keys.add(edge_key)
            theme = int(paper_node.get("theme", 0))
            current = strongest_anchor_by_theme.get(theme)
            if current is None or score > current[0]:
                strongest_anchor_by_theme[theme] = (score, anchor["label"])
            edges.append(
                {
                    "source": paper_node["id"],
                    "target": anchor["id"],
                    "weight": score,
                    "type": "concept_anchor",
                    "retained_by_floor": retained_by_floor,
                }
            )

        for paper_pos, paper_node in enumerate(nodes):
            scored: list[tuple[float, dict]] = []
            for anchor in anchors:
                key = anchor["key"]
                anchor_pos = concept_keys.index(key)
                score = float(concept_sims[paper_pos, anchor_pos])
                if score >= concept_similarity_threshold:
                    scored.append((score, anchor))
            scored.sort(key=lambda item: item[0], reverse=True)
            for score, anchor in scored[:concept_top_k]:
                add_concept_edge(paper_pos, paper_node, anchor, score)

        for paper_pos, paper_node in enumerate(nodes):
            for anchor in anchors:
                if anchor.get("retention_count", 0) < 2:
                    continue
                anchor_pos = concept_keys.index(anchor["key"])
                score = float(concept_sims[paper_pos, anchor_pos])
                if score >= CONCEPT_RETENTION_THRESHOLD:
                    add_concept_edge(
                        paper_pos,
                        paper_node,
                        anchor,
                        score,
                        retained_by_floor=score < concept_similarity_threshold,
                    )

    single_concepts_by_paper: dict[str, list[str]] = {}
    kept_anchors: list[dict] = []
    kept_anchor_ids: set[str] = set()
    for anchor in anchors:
        connected = [
            edge
            for edge in edges
            if edge["type"] == "concept_anchor" and edge["target"] == anchor["id"]
        ]
        if len(connected) > 1:
            kept_anchors.append(anchor)
            kept_anchor_ids.add(anchor["id"])
        elif len(connected) == 1:
            source_id = connected[0]["source"]
            single_concepts_by_paper.setdefault(source_id, []).append(anchor["label"])

    if single_concepts_by_paper:
        for node in nodes:
            concepts = single_concepts_by_paper.get(node["id"], [])
            if concepts:
                node["single_concepts"] = sorted(set(concepts))
    anchors = kept_anchors
    edges = [
        edge
        for edge in edges
        if edge["type"] != "concept_anchor" or edge["target"] in kept_anchor_ids
    ]
    nodes.extend(anchors)

    paper_nodes = [node for node in nodes if node["type"] == "paper"]
    for anchor_pos, anchor in enumerate(anchors):
        connected = [
            edge for edge in edges if edge["type"] == "concept_anchor" and edge["target"] == anchor["id"]
        ]
        if connected:
            coords = []
            weights = []
            for edge in connected:
                paper = next(node for node in paper_nodes if node["id"] == edge["source"])
                coords.append([paper["x"], paper["y"]])
                weights.append(max(float(edge["weight"]), 1e-6))
            coords_arr = np.asarray(coords, dtype=np.float32)
            weights_arr = np.asarray(weights, dtype=np.float32)
            weights_arr = weights_arr / np.maximum(weights_arr.sum(), 1e-12)
            coord = (coords_arr * weights_arr[:, None]).sum(axis=0)
        else:
            coord = np.array([0.0, 0.0], dtype=np.float32)
        direction = coord / np.maximum(np.linalg.norm(coord), 1e-12)
        if np.linalg.norm(direction) < 1e-9:
            angle = 2.0 * np.pi * anchor_pos / max(len(anchors), 1)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        anchor_coord = coord + (direction * 0.24)
        anchor["x"] = float(anchor_coord[0])
        anchor["y"] = float(anchor_coord[1])

    theme_names = {
        theme_id: label
        for theme_id, (_score, label) in strongest_anchor_by_theme.items()
    }
    for node in nodes:
        if node.get("type") == "paper":
            theme = int(node.get("theme", 0))
            node["theme_label"] = theme_names.get(theme, f"Theme {theme + 1}")
        node.pop("vector", None)
        node.pop("matrix_pos", None)
        node.pop("retention_count", None)

    by_pair: dict[tuple[str, str, str], dict] = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key not in by_pair or edge["weight"] > by_pair[key]["weight"]:
            by_pair[key] = edge

    summary = _paper_summary(nodes, custom_paper_sims)
    summary["theme_count"] = len(set(theme_labels))
    summary["theme_labels"] = [
        theme_names.get(theme_id, f"Theme {theme_id + 1}")
        for theme_id in sorted(set(theme_labels))
    ]
    summary["position_source"] = position_source

    final_edges = list(by_pair.values())
    node_lookup = {node["id"]: node for node in nodes}
    for node in nodes:
        node["paper_connection_count"] = 0
        node["concept_connection_count"] = 0
        node["connected_paper_count"] = 0

    for edge in final_edges:
        source = node_lookup.get(edge["source"])
        target = node_lookup.get(edge["target"])
        if source is None or target is None:
            continue
        if edge["type"] == "paper_similarity":
            source["paper_connection_count"] += 1
            target["paper_connection_count"] += 1
        elif edge["type"] == "concept_anchor":
            if source["type"] == "paper":
                source["concept_connection_count"] += 1
            if target["type"] == "concept":
                target["connected_paper_count"] += 1

    return {
        "nodes": nodes,
        "edges": final_edges,
        "summary": summary,
        "counts": {
            "papers": len(paper_indices),
            "concepts": sum(1 for node in nodes if node["type"] == "concept"),
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

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_lookup = {node["id"]: node for node in nodes}

    fig = go.Figure()
    for edge_type in ("paper_similarity", "concept_anchor"):
        edge_group = [edge for edge in edges if edge.get("type") == edge_type]
        if not edge_group:
            continue
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        hover_x_values: list[float] = []
        hover_y_values: list[float] = []
        hover_text: list[str] = []
        for edge in edge_group:
            source = node_lookup.get(edge["source"])
            target = node_lookup.get(edge["target"])
            if source is None or target is None:
                continue
            x_values.extend([source["x"], target["x"], None])
            y_values.extend([source["y"], target["y"], None])
            distance = 1.0 - float(edge["weight"])
            source_label = escape(str(source.get("label", source["id"])))
            target_label = escape(str(target.get("label", target["id"])))
            if edge_type == "paper_similarity":
                hover_parts = [
                    f"<b>{source_label}</b> - <b>{target_label}</b>",
                    f"custom similarity: {edge['weight']:.3f}",
                    f"custom distance: {distance:.3f}",
                    f"embedding similarity: {edge.get('embedding_similarity', edge['weight']):.3f}",
                    f"category overlap: {edge.get('category_overlap', 0.0):.3f}",
                    f"concept overlap: {edge.get('concept_overlap', 0.0):.3f}",
                    f"keyword overlap: {edge.get('keyword_overlap', 0.0):.3f}",
                ]
                shared_categories = edge.get("shared_categories", [])
                shared_concepts = edge.get("shared_concepts", [])
                shared_keywords = edge.get("shared_keywords", [])
                if shared_categories:
                    hover_parts.append(
                        "shared categories: "
                        + escape(", ".join(str(item) for item in shared_categories))
                    )
                if shared_concepts:
                    hover_parts.append(
                        "shared concepts: "
                        + escape(", ".join(str(item) for item in shared_concepts))
                    )
                if shared_keywords:
                    hover_parts.append(
                        "shared keywords: "
                        + escape(", ".join(str(item) for item in shared_keywords))
                    )
                hover_label = "<br>".join(hover_parts)
            else:
                hover_parts = [
                    f"<b>{source_label}</b> - <b>{target_label}</b>",
                    f"similarity: {edge['weight']:.3f}",
                    f"distance: {distance:.3f}",
                ]
                if edge.get("retained_by_floor"):
                    hover_parts.append(
                        f"retained at concept floor: {CONCEPT_RETENTION_THRESHOLD:.1f}"
                    )
                hover_label = "<br>".join(hover_parts)
            for fraction in (0.4, 0.5, 0.6):
                hover_x_values.append(
                    source["x"] + (target["x"] - source["x"]) * fraction
                )
                hover_y_values.append(
                    source["y"] + (target["y"] - source["y"]) * fraction
                )
                hover_text.append(hover_label)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=edge_type.replace("_", " "),
                line=dict(
                    width=2 if edge_type == "paper_similarity" else 1.5,
                    color={
                        "paper_similarity": "rgba(37, 99, 235, 0.42)",
                        "concept_anchor": "rgba(168, 85, 247, 0.34)",
                    }[edge_type],
                ),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hover_x_values,
                y=hover_y_values,
                mode="markers",
                name=f"{edge_type.replace('_', ' ')} similarity",
                marker=dict(size=14, color="rgba(0, 0, 0, 0)"),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    paper_nodes = [node for node in nodes if node.get("type") == "paper"]
    if paper_nodes:
        fig.add_trace(
            go.Scatter(
                x=[node["x"] for node in paper_nodes],
                y=[node["y"] for node in paper_nodes],
                mode="markers+text",
                name="Paper",
                text=[node["label"] for node in paper_nodes],
                textposition="top center",
                marker=dict(
                    symbol="circle",
                    size=19,
                    color="#2563eb",
                    line=dict(width=1.5, color="white"),
                ),
                customdata=[
                    [
                        node.get("title", node["label"]),
                        node.get("primary_category", "unknown"),
                        node.get("paper_connection_count", 0),
                        node.get("concept_connection_count", 0),
                        ", ".join(node.get("single_concepts", [])) or "none",
                    ]
                    for node in paper_nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "category: %{customdata[1]}<br>"
                    "paper links: %{customdata[2]}<br>"
                    "shared concept links: %{customdata[3]}<br>"
                    "paper-only concepts: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    concept_nodes = [node for node in nodes if node.get("type") == "concept"]
    if concept_nodes:
        fig.add_trace(
            go.Scatter(
                x=[node["x"] for node in concept_nodes],
                y=[node["y"] for node in concept_nodes],
                mode="markers+text",
                name="Concept",
                text=[node["label"] for node in concept_nodes],
                textposition="bottom center",
                marker=dict(
                    symbol="square",
                    size=15,
                    color="#a855f7",
                    line=dict(width=1.5, color="white"),
                ),
                customdata=[
                    [
                        node.get("title", node["label"]),
                        node.get("connected_paper_count", 0),
                    ]
                    for node in concept_nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "type: concept<br>"
                    "connected papers: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Workspace Concept Map", y=0.98, yanchor="top"),
        template="plotly_white",
        height=760,
        margin=dict(l=18, r=18, t=92, b=18),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(visible=False, zeroline=False)
    fig.update_yaxes(visible=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig
