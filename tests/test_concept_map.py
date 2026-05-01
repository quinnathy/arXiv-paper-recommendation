from types import SimpleNamespace

import numpy as np

from recommender.concept_map import build_workspace_concept_map


def _unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_workspace_map_contains_only_positioned_paper_nodes():
    first = _unit([1.0, 0.0, 0.0])
    second = _unit([0.0, 1.0, 0.0])
    index = SimpleNamespace(
        embeddings=np.stack([first, second]).astype(np.float32),
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
    assert graph["counts"] == {"papers": 2, "concepts": 0}
    assert [node["type"] for node in graph["nodes"]] == ["paper", "paper"]
    assert [node["key"] for node in graph["nodes"]] == ["2401.00001", "2401.00002"]
    assert all(isinstance(node["x"], float) for node in graph["nodes"])
    assert all(isinstance(node["y"], float) for node in graph["nodes"])
    assert graph["summary"]["position_source"] == "fallback"


def test_workspace_map_ignores_unknown_workspace_ids():
    index = SimpleNamespace(
        embeddings=np.stack([_unit([1.0, 0.0])]).astype(np.float32),
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
    assert [node["key"] for node in graph["nodes"]] == ["2401.00001"]
