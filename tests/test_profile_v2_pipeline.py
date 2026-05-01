"""Higher-level onboarding/profile initialization diagnostics.

These tests intentionally print pipeline state under ``pytest -s``.  They use
the production registries, seed factories, free-text expansion helper, and
threshold initializer; the only test double is a deterministic embedding model
for fast free-text tests so the suite never needs network access.

To run the test:
    pytest -s tests/test_profile_v2_pipeline.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.concept_tags import (
    CONCEPT_TAG_MAP,
    load_concept_embedding_artifacts,
)
from pipeline.interest_expander import embed_free_text_interests, expand_interest
from ui.components import TOPIC_LABELS, available_topic_labels, expand_topic_labels
from user.profile import (
    MAX_THREADS,
    MERGE_THRESHOLD,
    ProfileInitializationResult,
    init_user_profile_v2,
    make_category_seed,
    make_concept_seed,
    make_freetext_seed,
)


DATA_DIR = Path(os.getenv("PROFILE_TEST_DATA_DIR", "data"))


@dataclass(frozen=True)
class TagChoice:
    label: str
    kind: str
    key: str


def _get(seed: Any, name: str, default: Any = None) -> Any:
    if isinstance(seed, dict):
        return seed.get(name, default)
    return getattr(seed, name, default)


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return (v / max(float(np.linalg.norm(v)), eps)).astype(np.float32)


def _unit_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


def effective_weight(seed: Any) -> float | None:
    """Best-effort effective seed weight for object/dataclass/dict seeds."""
    weight = _get(seed, "weight")
    reliability = _get(seed, "reliability")
    specificity = _get(seed, "specificity")
    if weight is not None and reliability is not None and specificity is not None:
        return float(weight) * float(reliability) * float(specificity)
    if weight is not None:
        return float(weight)
    return None


def seed_label(seed: Any) -> str:
    return str(
        _get(seed, "label")
        or _get(seed, "key")
        or _get(seed, "text")
        or _get(seed, "phrase")
        or seed
    )


def seed_source(seed: Any) -> str | None:
    return _get(seed, "source") or _get(seed, "kind") or _get(seed, "type")


def seed_vector(seed: Any) -> np.ndarray:
    vector = _get(seed, "vector", None)
    if vector is None:
        vector = _get(seed, "embedding", None)
    if vector is None:
        raise AssertionError(f"Seed has no vector/embedding: {seed!r}")
    return np.asarray(vector, dtype=np.float32)


def assert_unit_norm_rows(X: np.ndarray, atol: float = 1e-4) -> None:
    X = np.asarray(X)
    assert X.ndim == 2, f"Expected a 2D matrix, got shape {X.shape}"
    assert np.isfinite(X).all(), "Matrix contains non-finite values"
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=atol), norms


def print_seed_table(seeds: list[Any]) -> None:
    print("\nSEEDS")
    print(
        "idx | label | source | weight | reliability | specificity | "
        "split_power | effective_weight | vector_norm"
    )
    for i, seed in enumerate(seeds):
        vector = seed_vector(seed)
        print(
            f"{i:>3} | {seed_label(seed)} | {seed_source(seed)} | "
            f"{_get(seed, 'weight', None)} | {_get(seed, 'reliability', None)} | "
            f"{_get(seed, 'specificity', None)} | {_get(seed, 'split_power', None)} | "
            f"{effective_weight(seed)} | {np.linalg.norm(vector):.6f}"
        )


def print_pairwise_distances(seeds: list[Any]) -> None:
    X = _unit_rows(np.stack([seed_vector(seed) for seed in seeds]))
    D = 1.0 - X @ X.T
    rows: list[tuple[float, int, int]] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            rows.append((float(D[i, j]), i, j))
    print("\nPAIRWISE SEED DISTANCES")
    for dist, i, j in sorted(rows):
        print(
            f"{dist:.4f} | {i}:{seed_label(seeds[i])} [{seed_source(seeds[i])}] "
            f"<-> {j}:{seed_label(seeds[j])} [{seed_source(seeds[j])}]"
        )


def print_threads(
    seeds: list[Any],
    labels: np.ndarray | list[int] | None,
    centroids: np.ndarray,
    thread_weights: np.ndarray | list[float] | None = None,
) -> None:
    print("\nTHREADS")
    print(f"inferred k_u: {centroids.shape[0]}")
    if thread_weights is not None:
        print(f"thread_weights: {np.asarray(thread_weights, dtype=float).round(4).tolist()}")
    if labels is None:
        print("seed labels not exposed")
        return
    labels_arr = np.asarray(labels)
    for thread_id in range(centroids.shape[0]):
        assigned = np.where(labels_arr == thread_id)[0].tolist()
        names = [f"{seed_label(seeds[i])} [{seed_source(seeds[i])}]" for i in assigned]
        print(f"thread {thread_id}: {names}")


def _all_pair_distances_below(seeds: list[Any], threshold: float) -> bool:
    X = _unit_rows(np.stack([seed_vector(seed) for seed in seeds]))
    D = 1.0 - X @ X.T
    off_diag = D[~np.eye(D.shape[0], dtype=bool)]
    return bool(np.all(off_diag <= threshold + 1e-6))


def get_profile_outputs(
    profile_or_tuple: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Any | None]:
    """Normalize initializer/profile return formats for diagnostics."""
    if isinstance(profile_or_tuple, ProfileInitializationResult):
        return (
            np.asarray(profile_or_tuple.centroids),
            np.asarray(profile_or_tuple.seed_labels),
            np.asarray(profile_or_tuple.thread_weights),
            profile_or_tuple,
        )
    if hasattr(profile_or_tuple, "centroids"):
        labels = getattr(profile_or_tuple, "seed_labels", None)
        weights = getattr(profile_or_tuple, "thread_weights", None)
        return (
            np.asarray(profile_or_tuple.centroids),
            None if labels is None else np.asarray(labels),
            None if weights is None else np.asarray(weights),
            profile_or_tuple,
        )
    if isinstance(profile_or_tuple, tuple):
        centroids = np.asarray(profile_or_tuple[0])
        labels = np.asarray(profile_or_tuple[1]) if len(profile_or_tuple) > 1 else None
        weights = np.asarray(profile_or_tuple[2]) if len(profile_or_tuple) > 2 else None
        profile = profile_or_tuple[3] if len(profile_or_tuple) > 3 else None
        return centroids, labels, weights, profile
    return np.asarray(profile_or_tuple), None, None, None


def _canonical(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_tag(
    registry: dict[str, TagChoice],
    candidates: list[str],
    *,
    kind: str | None = None,
) -> TagChoice | None:
    """Find a tag by label/key/code, case-insensitively."""
    choices = [choice for choice in registry.values() if kind is None or choice.kind == kind]
    for candidate in candidates:
        c = _canonical(candidate)
        for choice in choices:
            if c in {
                _canonical(choice.label),
                _canonical(choice.key),
            }:
                return choice
    for candidate in candidates:
        c = _canonical(candidate)
        for choice in choices:
            if c and (c in _canonical(choice.label) or c in _canonical(choice.key)):
                return choice
    return None


def load_category_centroids_or_skip() -> dict[str, np.ndarray]:
    path = DATA_DIR / "category_centroids.npy"
    if not path.exists():
        pytest.skip(f"Missing category centroid artifact: {path}")
    return np.load(path, allow_pickle=True).item()


def load_concept_embeddings_or_skip() -> dict[str, np.ndarray]:
    try:
        return load_concept_embedding_artifacts(DATA_DIR)
    except FileNotFoundError as exc:
        pytest.skip(f"Missing concept embedding artifact: {exc}")


def build_unified_tag_pool(
    category_centroids: dict[str, np.ndarray],
    concept_embeddings: dict[str, np.ndarray],
) -> dict[str, TagChoice]:
    label_map: dict[str, TagChoice] = {}
    for key, tag in CONCEPT_TAG_MAP.items():
        if key in concept_embeddings:
            label_map[tag.label] = TagChoice(tag.label, "concept", key)
    for label in available_topic_labels(TOPIC_LABELS, category_centroids):
        if label not in label_map:
            label_map[label] = TagChoice(label, "category", label)
    return label_map


class SemanticTestEmbeddingModel:
    """Deterministic free-text embedder backed by real local artifacts."""

    def __init__(
        self,
        concept_embeddings: dict[str, np.ndarray],
        category_centroids: dict[str, np.ndarray],
    ) -> None:
        self.concept_embeddings = concept_embeddings
        self.category_centroids = category_centroids
        self.embedding_dim = next(iter(concept_embeddings.values())).shape[0]

    def _vectors_for_text(self, text: str) -> list[np.ndarray]:
        t = text.lower()
        vectors: list[np.ndarray] = []
        if any(word in t for word in ["single-cell", "genomics", "bioinformatics", "biology"]):
            vectors.append(self.concept_embeddings["computational_biology"])
        if "medical imaging" in t or "radiology" in t or "diagnosis" in t:
            vectors.append(self.concept_embeddings["medical_imaging"])
            vectors.append(self.concept_embeddings["healthcare_ai"])
        if "clinical" in t or "healthcare" in t:
            vectors.append(self.concept_embeddings["healthcare_ai"])
        if "robot" in t or "navigation" in t or "embodied" in t:
            vectors.append(self.concept_embeddings["robotics_embodied"])
            vectors.append(self.concept_embeddings["reinforcement_learning"])
            if "cs.RO" in self.category_centroids:
                vectors.append(self.category_centroids["cs.RO"])
        if "weather" in t or "climate" in t or "forecasting" in t:
            vectors.append(self.concept_embeddings["climate_weather"])
            vectors.append(self.concept_embeddings["scientific_ml"])
        if "llm" in t or "language model" in t:
            vectors.append(self.concept_embeddings["llms"])
        if "machine learning" in t:
            for code in ("cs.LG", "stat.ML"):
                if code in self.category_centroids:
                    vectors.append(self.category_centroids[code])
        return vectors

    def embed_papers(self, papers: list[dict]) -> np.ndarray:
        rows = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            vectors = self._vectors_for_text(text)
            if not vectors:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                rows.append(_unit(rng.standard_normal(self.embedding_dim)))
            else:
                rows.append(_unit(np.mean(np.stack(vectors), axis=0)))
        return np.stack(rows).astype(np.float32)


def build_onboarding_seeds(
    tags: list[TagChoice],
    free_texts: list[str] | None = None,
    *,
    category_centroids: dict[str, np.ndarray],
    concept_embeddings: dict[str, np.ndarray],
    model: object | None = None,
) -> list[Any]:
    seeds: list[Any] = []
    for tag in tags:
        if tag.kind == "category":
            category_codes = expand_topic_labels(
                [tag.label],
                TOPIC_LABELS,
                category_centroids,
            )
            if not category_codes:
                raise AssertionError(f"No available categories for tag: {tag}")
            for code in category_codes:
                seeds.append(
                    make_category_seed(code, tag.label, category_centroids[code])
                )
        elif tag.kind == "concept":
            concept = CONCEPT_TAG_MAP[tag.key]
            seeds.append(
                make_concept_seed(
                    tag.key,
                    concept.label,
                    concept_embeddings[tag.key],
                )
            )
        else:
            raise AssertionError(f"Unknown tag kind: {tag}")

    if free_texts:
        if model is None:
            model = SemanticTestEmbeddingModel(concept_embeddings, category_centroids)
        for phrase, embedding in embed_free_text_interests(free_texts, model):
            seeds.append(make_freetext_seed(phrase, embedding))
    return seeds


def default_merge_threshold() -> float:
    return float(os.getenv("PROFILE_TEST_MERGE_THRESHOLD", str(MERGE_THRESHOLD)))


def default_max_threads() -> int:
    return int(os.getenv("PROFILE_TEST_MAX_THREADS", str(MAX_THREADS)))


def init_and_unpack(
    seeds: list[Any],
    *,
    merge_threshold: float | None = None,
    max_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Any | None]:
    result = init_user_profile_v2(
        seeds,
        max_threads=default_max_threads() if max_threads is None else max_threads,
        merge_threshold=default_merge_threshold() if merge_threshold is None else merge_threshold,
    )
    return get_profile_outputs(result)


def _loaded_context() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, TagChoice]]:
    category_centroids = load_category_centroids_or_skip()
    concept_embeddings = load_concept_embeddings_or_skip()
    pool = build_unified_tag_pool(category_centroids, concept_embeddings)
    return category_centroids, concept_embeddings, pool


def _require_tag(
    pool: dict[str, TagChoice],
    candidates: list[str],
    *,
    kind: str | None = None,
) -> TagChoice:
    tag = find_tag(pool, candidates, kind=kind)
    if tag is None:
        pytest.skip(f"Could not find tag matching {candidates}; available={sorted(pool)}")
    return tag


@pytest.mark.diagnostic
def test_unified_tag_pool_routes_category_and_concept_tags():
    category_centroids, concept_embeddings, pool = _loaded_context()
    ml = _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category")
    healthcare = _require_tag(pool, ["Healthcare AI", "healthcare_ai"], kind="concept")

    seeds = build_onboarding_seeds(
        [ml, healthcare],
        category_centroids=category_centroids,
        concept_embeddings=concept_embeddings,
    )
    sources = [seed_source(seed) or "" for seed in seeds]

    print(f"\nselected UI tags: {[ml.label, healthcare.label]}")
    print(f"backend seed sources: {sources}")
    print_seed_table(seeds)

    assert ml.label in pool
    assert healthcare.label in pool
    assert len(seeds) >= 2
    assert any("category" in source or "arxiv" in source for source in sources)
    assert any("tag" in source or "concept" in source or "predefined" in source for source in sources)
    assert_unit_norm_rows(np.stack([seed_vector(seed) for seed in seeds]))


@pytest.mark.diagnostic
def test_overlapping_ml_interests_form_one_thread():
    category_centroids, concept_embeddings, pool = _loaded_context()
    tags = [
        _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category"),
        _require_tag(pool, ["Artificial Intelligence", "cs.AI"], kind="category"),
        _require_tag(pool, ["Neural Networks", "cs.NE"], kind="category"),
    ]
    seeds = build_onboarding_seeds(
        tags,
        category_centroids=category_centroids,
        concept_embeddings=concept_embeddings,
    )

    centroids, labels, thread_weights, _ = init_and_unpack(seeds)
    k_u = centroids.shape[0]

    print(f"\nselected tags: {[tag.label for tag in tags]}")
    print_seed_table(seeds)
    print_pairwise_distances(seeds)
    print_threads(seeds, labels, centroids, thread_weights)

    assert 1 <= k_u <= default_max_threads()
    assert_unit_norm_rows(centroids)
    if thread_weights is not None:
        assert np.isclose(np.asarray(thread_weights).sum(), 1.0)
    if k_u != 1:
        assert k_u <= 2
        assert thread_weights is not None
        assert float(np.max(thread_weights)) >= 0.65


@pytest.mark.diagnostic
def test_ml_category_context_attaches_to_specific_free_text_thread():
    category_centroids, concept_embeddings, pool = _loaded_context()
    tags = [
        _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category"),
        _require_tag(pool, ["Artificial Intelligence", "cs.AI"], kind="category"),
    ]
    free_texts = ["single-cell perturbation modeling"]
    seeds = build_onboarding_seeds(
        tags,
        free_texts,
        category_centroids=category_centroids,
        concept_embeddings=concept_embeddings,
    )
    centroids, labels, thread_weights, _ = init_and_unpack(seeds)

    print(f"\nselected tags: {[tag.label for tag in tags]}")
    print(f"free-text: {free_texts}")
    print(f"expanded free-text: {[expand_interest(text) for text in free_texts]}")
    print_seed_table(seeds)
    print_pairwise_distances(seeds)
    print_threads(seeds, labels, centroids, thread_weights)

    assert_unit_norm_rows(centroids)
    assert 1 <= centroids.shape[0] <= default_max_threads()
    free_text_indices = [i for i, seed in enumerate(seeds) if seed_source(seed) == "free_text"]
    assert free_text_indices, "Expected a free-text seed"
    if centroids.shape[0] != 1:
        assert centroids.shape[0] <= 2
        assert labels is not None and thread_weights is not None
        ft_thread = int(np.asarray(labels)[free_text_indices[0]])
        assert float(np.asarray(thread_weights)[ft_thread]) >= 0.45
        category_threads = {
            int(np.asarray(labels)[i])
            for i, seed in enumerate(seeds)
            if seed_source(seed) == "arxiv_category"
        }
        assert len(category_threads) <= 1


@pytest.mark.diagnostic
def test_healthcare_and_robotics_form_two_threads():
    category_centroids, concept_embeddings, pool = _loaded_context()
    healthcare = _require_tag(pool, ["Healthcare AI", "healthcare_ai"], kind="concept")
    robotics = _require_tag(pool, ["Robotics", "cs.RO"], kind="category")
    free_texts = [
        "diffusion models for medical imaging",
        "reinforcement learning for robot navigation",
    ]
    seeds = build_onboarding_seeds(
        [healthcare, robotics],
        free_texts,
        category_centroids=category_centroids,
        concept_embeddings=concept_embeddings,
    )
    centroids, labels, thread_weights, _ = init_and_unpack(seeds)
    labels_arr = np.asarray(labels) if labels is not None else None

    print(f"\nselected tags: {[healthcare.label, robotics.label]}")
    print_seed_table(seeds)
    print_pairwise_distances(seeds)
    print_threads(seeds, labels, centroids, thread_weights)

    assert_unit_norm_rows(centroids)
    strongly_indicated_collapse = _all_pair_distances_below(seeds, default_merge_threshold())
    if centroids.shape[0] == 1 and strongly_indicated_collapse:
        print(
            "WARNING: healthcare/robotics collapsed because every seed pair is "
            f"within merge_threshold={default_merge_threshold():.3f}."
        )
    else:
        assert 2 <= centroids.shape[0] <= 3
    if labels_arr is not None:
        by_label = {seed_label(seed): i for i, seed in enumerate(seeds)}
        assert labels_arr[by_label["Healthcare AI"]] == labels_arr[
            by_label["diffusion models for medical imaging"]
        ]
        assert labels_arr[by_label["Robotics"]] == labels_arr[
            by_label["reinforcement learning for robot navigation"]
        ]
        unique_threads = set(labels_arr.tolist())
        if len(unique_threads) == 1:
            print("WARNING: healthcare and robotics collapsed into one thread")
        assert len(unique_threads) > 1 or strongly_indicated_collapse, (
            "All four seeds collapsed into one default-threshold thread without "
            "all pairwise embeddings being below the merge threshold"
        )


@pytest.mark.diagnostic
def test_initializer_enforces_max_threads_on_many_distant_interests():
    category_centroids, concept_embeddings, pool = _loaded_context()
    tag_specs = [
        (["Healthcare AI", "healthcare_ai"], "concept"),
        (["Robotics", "cs.RO"], "category"),
        (["Climate & Weather", "climate_weather"], "concept"),
        (["Computer Vision", "cs.CV"], "category"),
        (["Computational Biology", "computational_biology"], "concept"),
    ]
    tags = [_require_tag(pool, candidates, kind=kind) for candidates, kind in tag_specs]
    free_texts = [
        "single-cell perturbation modeling",
        "reinforcement learning for robot navigation",
        "neural operators for weather forecasting",
    ]
    seeds = build_onboarding_seeds(
        tags,
        free_texts,
        category_centroids=category_centroids,
        concept_embeddings=concept_embeddings,
    )
    centroids, labels, thread_weights, _ = init_and_unpack(seeds, max_threads=3)

    print(f"\nselected tags: {[tag.label for tag in tags]}")
    print_seed_table(seeds)
    print_pairwise_distances(seeds)
    print_threads(seeds, labels, centroids, thread_weights)

    assert 1 <= centroids.shape[0] <= 3
    assert_unit_norm_rows(centroids)
    if thread_weights is not None:
        assert len(thread_weights) == centroids.shape[0]
        assert np.isclose(np.asarray(thread_weights).sum(), 1.0)
    if labels is not None:
        assert len(labels) == len(seeds)


@pytest.mark.diagnostic
def test_threshold_sweep_on_canonical_onboarding_cases():
    category_centroids, concept_embeddings, pool = _loaded_context()
    thresholds = [
        float(x)
        for x in os.getenv(
            "PROFILE_TEST_THRESHOLD_SWEEP",
            "0.15,0.18,0.20,0.22,0.25,0.28,0.30",
        ).split(",")
    ]
    max_threads = int(os.getenv("PROFILE_TEST_SWEEP_MAX_THREADS", "3"))

    cases = [
        (
            "overlapping_ml",
            [
                _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category"),
                _require_tag(pool, ["Artificial Intelligence", "cs.AI"], kind="category"),
                _require_tag(pool, ["Neural Networks", "cs.NE"], kind="category"),
            ],
            [],
            (1, 2),
        ),
        (
            "ml_healthcare",
            [
                _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category"),
                _require_tag(pool, ["Healthcare AI", "healthcare_ai"], kind="concept"),
            ],
            ["diffusion models for medical imaging"],
            (1, 2),
        ),
        (
            "healthcare_robotics",
            [
                _require_tag(pool, ["Healthcare AI", "healthcare_ai"], kind="concept"),
                _require_tag(pool, ["Robotics", "cs.RO"], kind="category"),
            ],
            [
                "diffusion models for medical imaging",
                "reinforcement learning for robot navigation",
            ],
            (2, 3),
        ),
        (
            "category_context_plus_specific_bio",
            [
                _require_tag(pool, ["Artificial Intelligence", "cs.AI"], kind="category"),
                _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category"),
            ],
            ["single-cell perturbation modeling"],
            (1, 2),
        ),
    ]

    print("\nTHRESHOLD SWEEP")
    print("case | threshold | k_u | weights | assignments")
    for case_name, tags, free_texts, expected_default_range in cases:
        seeds = build_onboarding_seeds(
            tags,
            free_texts,
            category_centroids=category_centroids,
            concept_embeddings=concept_embeddings,
        )
        default_k = None
        for threshold in thresholds:
            centroids, labels, thread_weights, _ = init_and_unpack(
                seeds,
                merge_threshold=threshold,
                max_threads=max_threads,
            )
            assignments = (
                "n/a"
                if labels is None
                else ", ".join(f"{seed_label(seed)}->{int(labels[i])}" for i, seed in enumerate(seeds))
            )
            weights = None if thread_weights is None else np.asarray(thread_weights).round(3).tolist()
            print(f"{case_name} | {threshold:.2f} | {centroids.shape[0]} | {weights} | {assignments}")
            assert 1 <= centroids.shape[0] <= max_threads
            assert_unit_norm_rows(centroids)
            if abs(threshold - MERGE_THRESHOLD) < 1e-9:
                default_k = centroids.shape[0]
        if default_k is None:
            centroids, _, _, _ = init_and_unpack(
                seeds,
                merge_threshold=MERGE_THRESHOLD,
                max_threads=max_threads,
            )
            default_k = centroids.shape[0]
        lo, hi = expected_default_range
        if not (lo <= default_k <= hi):
            assert _all_pair_distances_below(seeds, MERGE_THRESHOLD), (
                f"{case_name} default k={default_k} outside expected range "
                f"{expected_default_range}"
            )


@pytest.mark.diagnostic
def test_print_full_cold_start_profile_pipeline():
    category_centroids, concept_embeddings, pool = _loaded_context()
    tags = [
        _require_tag(pool, ["Machine Learning", "cs.LG"], kind="category"),
        _require_tag(pool, ["Healthcare AI", "healthcare_ai"], kind="concept"),
        _require_tag(pool, ["Computational Biology", "computational_biology"], kind="concept"),
    ]
    free_texts = [
        "single-cell perturbation modeling",
        "LLMs for clinical decision support",
    ]
    merge_threshold = default_merge_threshold()
    max_threads = default_max_threads()
    seeds = build_onboarding_seeds(
        tags,
        free_texts,
        category_centroids=category_centroids,
        concept_embeddings=concept_embeddings,
    )
    result = init_user_profile_v2(
        seeds,
        merge_threshold=merge_threshold,
        max_threads=max_threads,
    )
    centroids, labels, thread_weights, profile = get_profile_outputs(result)

    print("\nFULL COLD-START PROFILE PIPELINE")
    print(f"user-facing selected tags: {[tag.label for tag in tags]}")
    print(f"free-text interests: {free_texts}")
    print(f"expanded free-text strings: {[expand_interest(text) for text in free_texts]}")
    print_seed_table(seeds)
    print_pairwise_distances(seeds)
    print(f"merge_threshold: {merge_threshold}")
    print(f"max_threads: {max_threads}")
    print(f"inferred k_u: {centroids.shape[0]}")
    print_threads(seeds, labels, centroids, thread_weights)
    if profile is not None and hasattr(profile, "thread_labels"):
        print(f"thread labels: {profile.thread_labels}")
    if thread_weights is not None:
        print(f"thread weights: {np.asarray(thread_weights).round(4).tolist()}")
    print(f"final centroid shape: {centroids.shape}")
    print(f"centroid norms: {np.linalg.norm(centroids, axis=1).round(6).tolist()}")

    assert centroids.size > 0
    assert 1 <= centroids.shape[0] <= max_threads
    assert centroids.shape == (centroids.shape[0], next(iter(concept_embeddings.values())).shape[0])
    assert_unit_norm_rows(centroids)
    if labels is not None:
        assert len(labels) == len(seeds)
    if thread_weights is not None:
        assert np.isclose(np.asarray(thread_weights).sum(), 1.0)
