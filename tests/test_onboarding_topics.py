"""Tests for onboarding topic label expansion."""

from __future__ import annotations

import numpy as np
import pytest

from ui.components import (
    MAX_ONBOARDING_TAGS,
    available_topic_labels,
    build_onboarding_tag_pill_rules,
    expand_topic_labels,
    trim_onboarding_tag_selection,
)
from ui.onboarding import make_category_seeds_from_topic_labels
from user.profile import init_user_profile_v2


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _centroids(codes: list[str], dim: int = 8) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for i, code in enumerate(codes):
        v = np.zeros(dim, dtype=np.float32)
        v[i % dim] = 1.0
        centroids[code] = v
    return centroids


def test_one_label_expands_to_multiple_available_categories():
    topic_labels = {"Healthcare AI": ["cs.LG", "cs.CV", "cs.CL"]}
    category_centroids = _centroids(["cs.LG", "cs.CV", "cs.CL"])

    assert expand_topic_labels(
        ["Healthcare AI"],
        topic_labels,
        category_centroids,
    ) == ["cs.LG", "cs.CV", "cs.CL"]


def test_missing_categories_are_skipped():
    topic_labels = {"Healthcare AI": ["cs.LG", "cs.CV", "q-bio.QM"]}
    category_centroids = _centroids(["cs.LG", "cs.CV"])

    assert expand_topic_labels(
        ["Healthcare AI"],
        topic_labels,
        category_centroids,
    ) == ["cs.LG", "cs.CV"]


def test_expansion_deduplicates_across_labels_preserving_order():
    topic_labels = {
        "Machine Learning": ["cs.LG", "stat.ML"],
        "Computer Vision": ["cs.CV", "cs.LG"],
    }
    category_centroids = _centroids(["cs.LG", "stat.ML", "cs.CV"])

    assert expand_topic_labels(
        ["Machine Learning", "Computer Vision"],
        topic_labels,
        category_centroids,
    ) == ["cs.LG", "stat.ML", "cs.CV"]


def test_unknown_labels_do_not_crash_expansion():
    topic_labels = {"Machine Learning": ["cs.LG", "stat.ML"]}
    category_centroids = _centroids(["cs.LG", "stat.ML"])

    assert expand_topic_labels(
        ["Unknown Label", "Machine Learning"],
        topic_labels,
        category_centroids,
    ) == ["cs.LG", "stat.ML"]


def test_available_topic_labels_include_labels_with_any_available_category():
    topic_labels = {
        "Machine Learning": ["cs.LG", "stat.ML"],
        "Astrophysics": ["astro-ph.CO"],
        "Unavailable Topic": ["fake.CAT"],
    }
    category_centroids = _centroids(["cs.LG", "astro-ph.CO"])

    assert available_topic_labels(topic_labels, category_centroids) == [
        "Machine Learning",
        "Astrophysics",
    ]


def test_empty_expansion_produces_clear_onboarding_error():
    category_centroids = _centroids(["cs.LG"])

    with pytest.raises(ValueError, match="selected onboarding topics"):
        make_category_seeds_from_topic_labels(
            ["Unavailable Topic"],
            category_centroids,
        )


def test_onboarding_tag_selection_is_trimmed_to_limit():
    selected = ["A", "B", "C", "D", "E"]

    assert trim_onboarding_tag_selection(selected) == selected[:MAX_ONBOARDING_TAGS]


def test_onboarding_tag_selection_drops_unavailable_options_before_limit():
    selected = ["A", "Missing", "B", "C", "D"]

    assert trim_onboarding_tag_selection(
        selected,
        valid_options=["A", "B", "C", "D"],
    ) == ["A", "B", "C"]


def test_onboarding_tag_pill_rules_keep_unselected_clickable_below_limit():
    rules = build_onboarding_tag_pill_rules(
        options=["A", "B", "C", "D"],
        option_colors=["#111111", "#222222", "#333333", "#444444"],
        selected_labels=["A", "B"],
    )
    css = "".join(rules)

    assert "#444444" in css
    assert "pointer-events: none" not in css
    assert "#E5E7EB" not in css


def test_onboarding_tag_pill_rules_gray_only_unselected_tags_at_limit():
    rules = build_onboarding_tag_pill_rules(
        options=["A", "B", "C", "D"],
        option_colors=["#111111", "#222222", "#333333", "#444444"],
        selected_labels=["A", "B", "C"],
    )
    css = "".join(rules)

    assert "button:nth-child(4):not([aria-checked='true'])" in css
    assert "pointer-events: none" in rules[3]
    assert "#E5E7EB" in rules[3]
    assert "pointer-events: none" not in "".join(rules[:3])
    assert "#111111" in rules[0]


def test_onboarding_topic_expansion_pipeline_prints_diagnostics():
    selected_labels = ["Healthcare AI", "Machine Learning"]
    category_centroids = {
        "cs.LG": _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        "cs.CV": _unit(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        "cs.CL": _unit(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        "stat.ML": _unit(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
    }

    seeds = make_category_seeds_from_topic_labels(selected_labels, category_centroids)
    expanded_categories = [seed.label for seed in seeds]
    expanded_codes = expand_topic_labels(
        selected_labels,
        {
            "Healthcare AI": [
                "cs.LG",
                "cs.CV",
                "cs.CL",
                "cs.AI",
                "stat.ML",
                "q-bio.QM",
            ],
            "Machine Learning": ["cs.LG", "stat.ML"],
        },
        category_centroids,
    )
    requested = [
        "cs.LG",
        "cs.CV",
        "cs.CL",
        "cs.AI",
        "stat.ML",
        "q-bio.QM",
    ]
    skipped = [code for code in requested if code not in category_centroids]

    result = init_user_profile_v2(seeds)
    centroid_norms = np.linalg.norm(result.centroids, axis=1)

    print("\nONBOARDING TOPIC EXPANSION PIPELINE")
    print(f"selected human labels: {selected_labels}")
    print(f"expanded arXiv categories: {expanded_codes}")
    print(f"unavailable skipped categories: {skipped}")
    print(f"seed labels: {expanded_categories}")
    print(f"number of seed vectors: {len(seeds)}")
    print(f"output centroid shape: {result.centroids.shape}")
    print(f"centroid norms: {centroid_norms.round(6).tolist()}")

    assert expanded_codes == ["cs.LG", "cs.CV", "cs.CL", "stat.ML"]
    assert len(seeds) == len(expanded_codes)
    assert result.centroids.shape[0] >= 1
    assert np.allclose(centroid_norms, 1.0, atol=1e-6)
