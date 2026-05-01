from types import SimpleNamespace

import numpy as np

from recommender.concept_map import build_workspace_concept_map
from recommender.concept_map import make_workspace_concept_map_figure


def _unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_workspace_map_contains_only_cluster_centroid_and_positioned_paper_nodes():
    first = _unit([1.0, 0.0, 0.0])
    second = _unit([0.0, 1.0, 0.0])
    centroid = _unit([1.0, 1.0, 0.0])
    index = SimpleNamespace(
        embeddings=np.stack([first, second]).astype(np.float32),
        centroids=np.stack([centroid]).astype(np.float32),
        cluster_ids=np.array([0, 0], dtype=np.int32),
        paper_meta=[
            {
                "id": "2401.00001",
                "title": "First Paper",
                "abstract": "",
                "categories": "cs.LG",
            },
            {
                "id": "2401.00002",
                "title": "Second Paper",
                "abstract": "",
                "categories": "cs.CL",
            },
        ],
    )

    graph = build_workspace_concept_map(index, ["2401.00001", "2401.00002"])

    assert graph["edges"] == []
    assert graph["counts"] == {
        "papers": 2,
        "clusters": 1,
        "connections": 0,
        "concepts": 0,
    }
    assert [node["type"] for node in graph["nodes"]] == [
        "cluster_centroid",
        "paper",
        "paper",
    ]
    assert [node["key"] for node in graph["nodes"]] == [
        "0",
        "2401.00001",
        "2401.00002",
    ]
    assert graph["nodes"][1]["label"] == "A"
    assert graph["nodes"][0]["title"] == "cs.LG neighborhood (1 papers)"
    assert all(isinstance(node["x"], float) for node in graph["nodes"])
    assert all(isinstance(node["y"], float) for node in graph["nodes"])
    assert graph["summary"]["position_source"] == "fallback PCA"


def test_workspace_map_ignores_unknown_workspace_ids():
    index = SimpleNamespace(
        embeddings=np.stack([_unit([1.0, 0.0])]).astype(np.float32),
        centroids=np.stack([_unit([1.0, 0.0])]).astype(np.float32),
        cluster_ids=np.array([0], dtype=np.int32),
        paper_meta=[
            {
                "id": "2401.00001",
                "title": "Known Paper",
                "abstract": "",
                "categories": "cs.LG",
            }
        ],
    )

    graph = build_workspace_concept_map(index, ["missing", "2401.00001"])

    assert graph["counts"]["papers"] == 1
    assert [node["key"] for node in graph["nodes"]] == ["0", "2401.00001"]


def test_workspace_map_adds_ai_connections_between_paper_nodes():
    first = _unit([1.0, 0.0, 0.0])
    second = _unit([0.0, 1.0, 0.0])
    centroid = _unit([1.0, 1.0, 0.0])
    index = SimpleNamespace(
        embeddings=np.stack([first, second]).astype(np.float32),
        centroids=np.stack([centroid]).astype(np.float32),
        cluster_ids=np.array([0, 0], dtype=np.int32),
        paper_meta=[
            {"id": "2401.00001", "title": "First Paper", "categories": "cs.LG"},
            {"id": "2401.00002", "title": "Second Paper", "categories": "cs.CL"},
        ],
    )

    graph = build_workspace_concept_map(
        index,
        ["2401.00001", "2401.00002"],
        connections=[
            {
                "source": "2401.00001",
                "target": "2401.00002",
                "type": "methodology",
                "name": "Shared transformer",
                "rationale": "Both papers use transformer encoders.",
                "description": "Both papers use transformer encoders to model text.",
                "confidence": 0.9,
            },
            {
                "source": "2401.00001",
                "target": "missing",
                "type": "methodology",
                "name": "Invalid",
            },
        ],
    )

    assert graph["counts"]["connections"] == 1
    assert graph["edges"][0]["source"] == "paper:2401.00001"
    assert graph["edges"][0]["target"] == "paper:2401.00002"

    fig = make_workspace_concept_map_figure(graph)
    methodology_trace = next(trace for trace in fig.data if trace.name == "Methodology")
    assert methodology_trace.line.color == "#059669"
    assert "customdata[0]" in methodology_trace.hovertemplate
    assert "customdata[3]" in methodology_trace.hovertemplate
    assert methodology_trace.customdata[0][3] == "First Paper"
    assert methodology_trace.customdata[0][4] == "Second Paper"
    assert len([x for x in methodology_trace.x if x is not None]) > 3

    legend_names = {trace.name for trace in fig.data}
    assert {
        "Thesis",
        "Assumption",
        "Methodology",
        "Dataset",
        "Evaluation",
        "Multiple",
    }.issubset(legend_names)

    paper_trace = next(trace for trace in fig.data if trace.name == "Workspace paper")
    assert list(paper_trace.text) == ["A", "B"]
    assert paper_trace.textposition == "middle center"


def test_workspace_map_combines_multiple_connection_categories():
    first = _unit([1.0, 0.0, 0.0])
    second = _unit([0.0, 1.0, 0.0])
    centroid = _unit([1.0, 1.0, 0.0])
    index = SimpleNamespace(
        embeddings=np.stack([first, second]).astype(np.float32),
        centroids=np.stack([centroid]).astype(np.float32),
        cluster_ids=np.array([0, 0], dtype=np.int32),
        paper_meta=[
            {"id": "2401.00001", "title": "First Paper", "categories": "cs.LG"},
            {"id": "2401.00002", "title": "Second Paper", "categories": "cs.CL"},
        ],
    )

    graph = build_workspace_concept_map(
        index,
        ["2401.00001", "2401.00002"],
        connections=[
            {
                "source": "2401.00001",
                "target": "2401.00002",
                "type": "methodology",
                "name": "Shared method",
                "description": "Both papers use a shared method.",
            },
            {
                "source": "2401.00002",
                "target": "2401.00001",
                "type": "dataset",
                "name": "Shared dataset",
                "description": "Both papers evaluate on a related dataset.",
            },
        ],
    )

    assert graph["counts"]["connections"] == 1
    assert graph["edges"][0]["type"] == "multiple"
    assert graph["edges"][0]["connection_types"] == "Methodology, Dataset"
    assert "\n\n" in graph["edges"][0]["description"]

    fig = make_workspace_concept_map_figure(graph)
    multiple_trace = next(trace for trace in fig.data if trace.name == "Multiple")
    assert multiple_trace.line.color == "#111827"
    assert "Methodology, Dataset" in multiple_trace.customdata[0][1]
    assert "────────────────────" in multiple_trace.customdata[0][2]
