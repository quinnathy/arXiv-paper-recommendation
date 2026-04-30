from types import SimpleNamespace

import numpy as np

from recommender.concept_map import build_workspace_concept_map


def _unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_concepts_with_two_links_at_point_five_are_shown():
    concept = _unit([1.0, 0.0])
    near_concept = _unit([0.7, 0.714])
    index = SimpleNamespace(
        embeddings=np.stack([near_concept, near_concept]).astype(np.float32),
        paper_meta=[
            {
                "id": "2401.00001",
                "title": "First language model paper",
                "abstract": "",
                "categories": "cs.CL",
            },
            {
                "id": "2401.00002",
                "title": "Second language model paper",
                "abstract": "",
                "categories": "cs.CL",
            },
        ],
        concept_embeddings={"llms": concept},
    )

    graph = build_workspace_concept_map(
        index,
        ["2401.00001", "2401.00002"],
        concept_similarity_threshold=0.9,
        paper_similarity_threshold=0.9,
    )

    concept_nodes = [node for node in graph["nodes"] if node["type"] == "concept"]
    concept_edges = [
        edge for edge in graph["edges"] if edge["type"] == "concept_anchor"
    ]

    assert [node["key"] for node in concept_nodes] == ["llms"]
    assert len(concept_edges) == 2
    assert all(edge["retained_by_floor"] for edge in concept_edges)
    assert all(np.isclose(edge["weight"], float(near_concept @ concept)) for edge in concept_edges)


def test_concepts_with_fewer_than_two_point_five_links_are_hidden():
    concept = _unit([1.0, 0.0])
    near_concept = _unit([0.7, 0.714])
    far_concept = _unit([0.1, 0.995])
    index = SimpleNamespace(
        embeddings=np.stack([near_concept, far_concept]).astype(np.float32),
        paper_meta=[
            {
                "id": "2401.00001",
                "title": "First language model paper",
                "abstract": "",
                "categories": "cs.CL",
            },
            {
                "id": "2401.00002",
                "title": "Unrelated optimization paper",
                "abstract": "",
                "categories": "math.OC",
            },
        ],
        concept_embeddings={"llms": concept},
    )

    graph = build_workspace_concept_map(
        index,
        ["2401.00001", "2401.00002"],
        concept_similarity_threshold=0.5,
        paper_similarity_threshold=0.5,
    )

    assert [node for node in graph["nodes"] if node["type"] == "concept"] == []
