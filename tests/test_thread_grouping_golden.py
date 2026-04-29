"""Artifact-backed golden cases for threshold thread initialization."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from pipeline.concept_tags import (
    BROAD_CONCEPT_KEYS,
    CONCEPT_TAG_MAP,
    load_concept_embedding_artifacts,
)
from pipeline.interest_expander import embed_free_text_interests
from ui.components import TOPIC_LABELS, expand_topic_labels
from user.profile import (
    CORE_SPLIT_POWER,
    MAX_THREADS,
    MERGE_THRESHOLD,
    SeedSignal,
    initialize_user_centroids_threshold,
    make_category_seed,
    make_concept_seed,
    make_freetext_seed,
)


DATA_DIR = Path(os.getenv("PROFILE_TEST_DATA_DIR", "data"))


@dataclass(frozen=True)
class GoldenCase:
    name: str
    tags: tuple[str, ...]
    free_texts: tuple[str, ...]
    expected_k_range: tuple[int, int]


TAG_ALIASES: dict[str, tuple[str, str]] = {
    "machine_learning": ("category", "Machine Learning"),
    "deep_learning": ("category", "Deep Learning & Neural Networks"),
    "neural_networks": ("category", "Deep Learning & Neural Networks"),
    "large_language_models": ("concept", "llms"),
    "natural_language_processing": ("category", "Natural Language Processing"),
    "computer_vision": ("category", "Computer Vision"),
    "medical_imaging": ("concept", "medical_imaging"),
    "healthcare_ai": ("concept", "healthcare_ai"),
    "computational_biology": ("concept", "computational_biology"),
    "scientific_ml": ("concept", "scientific_ml"),
    "climate_weather": ("concept", "climate_weather"),
    "robotics": ("category", "Robotics"),
    "finance_economics_ai": ("concept", "financial_ml"),
    "artificial_intelligence": ("category", "Artificial Intelligence"),
    "mathematics": ("category", "Pure Mathematics"),
    "number_theory": ("category", "Pure Mathematics"),
    "statistics": ("category", "Statistics & Data Analysis"),
    "astrophysics": ("category", "Astrophysics & Cosmology"),
    "quantum_physics": ("category", "Quantum Information & Quantum Computing"),
}


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        "dense_ml_core",
        ("machine_learning", "deep_learning", "neural_networks"),
        (),
        (1, 1),
    ),
    GoldenCase(
        "ml_llm_nlp_rag",
        ("machine_learning", "large_language_models", "natural_language_processing"),
        ("retrieval augmented generation",),
        (1, 2),
    ),
    GoldenCase(
        "cv_medical_imaging_diffusion",
        ("computer_vision", "medical_imaging"),
        ("diffusion models for medical imaging",),
        (1, 2),
    ),
    GoldenCase(
        "ml_healthcare_ehr",
        ("machine_learning", "healthcare_ai"),
        ("clinical prediction with electronic health records",),
        (1, 1),
    ),
    GoldenCase(
        "ml_bio_single_cell",
        ("machine_learning", "computational_biology"),
        ("single-cell perturbation modeling",),
        (1, 2),
    ),
    GoldenCase(
        "scientific_ml_weather",
        ("scientific_ml", "climate_weather"),
        ("neural operators for weather forecasting",),
        (1, 2),
    ),
    GoldenCase(
        "healthcare_robotics",
        ("healthcare_ai", "robotics"),
        (
            "diffusion models for medical imaging",
            "reinforcement learning for robot navigation",
        ),
        (2, 2),
    ),
    GoldenCase(
        "bio_climate",
        ("computational_biology", "climate_weather"),
        ("single-cell perturbation modeling", "meteorological downscaling"),
        (2, 2),
    ),
    GoldenCase(
        "nlp_robotics",
        ("natural_language_processing", "robotics"),
        ("large language models for planning", "reinforcement learning for robot navigation"),
        (2, 2),
    ),
    GoldenCase(
        "cv_finance",
        ("computer_vision", "finance_economics_ai"),
        ("visual representation learning", "market forecasting with machine learning"),
        (2, 2),
    ),
    GoldenCase(
        "broad_ai_ml_specific_bio",
        ("artificial_intelligence", "machine_learning"),
        ("single-cell perturbation modeling",),
        (1, 1),
    ),
    GoldenCase(
        "math_ml_medical_diffusion",
        ("mathematics", "machine_learning"),
        ("diffusion models for medical imaging",),
        (1, 2),
    ),
    GoldenCase(
        "stats_ml_causal_healthcare",
        ("statistics", "machine_learning"),
        ("causal inference for healthcare",),
        (1, 1),
    ),
    GoldenCase(
        "ai_robotics_rl_navigation",
        ("artificial_intelligence", "robotics"),
        ("robot navigation with reinforcement learning",),
        (1, 2),
    ),
    GoldenCase("robotics_number_theory", ("robotics", "number_theory"), (), (2, 2)),
    GoldenCase("astrophysics_nlp", ("astrophysics", "natural_language_processing"), (), (2, 2)),
    GoldenCase("quantum_healthcare", ("quantum_physics", "healthcare_ai"), (), (2, 2)),
    GoldenCase(
        "climate_llm",
        ("climate_weather", "large_language_models"),
        (),
        (1, 2),
    ),
    GoldenCase(
        "healthcare_robotics_climate",
        ("healthcare_ai", "robotics", "climate_weather"),
        (
            "clinical prediction with electronic health records",
            "reinforcement learning for robot navigation",
            "neural operators for weather forecasting",
        ),
        (3, 3),
    ),
    GoldenCase(
        "bio_nlp_astrophysics",
        ("computational_biology", "natural_language_processing", "astrophysics"),
        (
            "single-cell perturbation modeling",
            "large language models for scientific text mining",
            "galaxy formation and cosmology",
        ),
        (3, 3),
    ),
    GoldenCase(
        "cv_finance_sciml",
        ("computer_vision", "finance_economics_ai", "scientific_ml"),
        (
            "visual representation learning",
            "market forecasting with machine learning",
            "neural operators for partial differential equations",
        ),
        (2, 3),
    ),
)


@pytest.fixture(scope="session")
def artifact_context() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    category_path = DATA_DIR / "category_centroids.npy"
    if not category_path.exists():
        pytest.skip(f"Missing category centroid artifact: {category_path}")
    try:
        concept_embeddings = load_concept_embedding_artifacts(DATA_DIR)
    except FileNotFoundError as exc:
        pytest.skip(f"Missing concept embedding artifact: {exc}")
    return np.load(category_path, allow_pickle=True).item(), concept_embeddings


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return (v / max(float(np.linalg.norm(v)), eps)).astype(np.float32)


class ArtifactBackedFreeTextModel:
    """Deterministic free-text embedder using only local embedding artifacts."""

    def __init__(
        self,
        concept_embeddings: dict[str, np.ndarray],
        category_centroids: dict[str, np.ndarray],
    ) -> None:
        self.concept_embeddings = concept_embeddings
        self.category_centroids = category_centroids
        self.embedding_dim = next(iter(concept_embeddings.values())).shape[0]

    def _maybe_concept(self, vectors: list[np.ndarray], key: str) -> None:
        if key in self.concept_embeddings:
            vectors.append(self.concept_embeddings[key])

    def _maybe_category(self, vectors: list[np.ndarray], code: str) -> None:
        if code in self.category_centroids:
            vectors.append(self.category_centroids[code])

    def _vectors_for_text(self, text: str) -> list[np.ndarray]:
        t = text.lower()
        vectors: list[np.ndarray] = []
        if any(term in t for term in ("single-cell", "genomics", "biology", "bioinformatics")):
            self._maybe_concept(vectors, "computational_biology")
        if any(term in t for term in ("medical imaging", "radiology", "clinical", "healthcare", "ehr", "electronic health")):
            self._maybe_concept(vectors, "healthcare_ai")
        if "medical imaging" in t or "diffusion" in t:
            self._maybe_concept(vectors, "medical_imaging")
            self._maybe_concept(vectors, "generative_models")
        if any(term in t for term in ("robot", "navigation", "embodied")):
            self._maybe_concept(vectors, "robotics_embodied")
            self._maybe_concept(vectors, "reinforcement_learning")
            self._maybe_category(vectors, "cs.RO")
        if any(term in t for term in ("weather", "climate", "meteorological", "forecasting")):
            self._maybe_concept(vectors, "climate_weather")
            self._maybe_concept(vectors, "scientific_ml")
        if any(term in t for term in ("neural operator", "partial differential", "pde")):
            self._maybe_concept(vectors, "scientific_ml")
        if any(term in t for term in ("language model", "llm", "retrieval augmented", "rag", "text mining")):
            self._maybe_concept(vectors, "llms")
            self._maybe_category(vectors, "cs.CL")
        if any(term in t for term in ("vision", "visual", "image")):
            self._maybe_category(vectors, "cs.CV")
        if any(term in t for term in ("finance", "market", "forecasting with machine learning")):
            self._maybe_concept(vectors, "financial_ml")
            self._maybe_concept(vectors, "time_series")
        if "causal" in t:
            self._maybe_concept(vectors, "causal_inference")
        if any(term in t for term in ("galaxy", "cosmology", "astrophysics")):
            self._maybe_category(vectors, "astro-ph.CO")
        if "quantum" in t:
            self._maybe_concept(vectors, "quantum_computing")
            self._maybe_category(vectors, "quant-ph")
        if "machine learning" in t:
            self._maybe_category(vectors, "cs.LG")
            self._maybe_category(vectors, "stat.ML")
        return vectors

    def embed_papers(self, papers: list[dict]) -> np.ndarray:
        rows = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            vectors = self._vectors_for_text(text)
            if vectors:
                rows.append(_unit(np.mean(np.stack(vectors), axis=0)))
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            rng = np.random.default_rng(seed)
            rows.append(_unit(rng.standard_normal(self.embedding_dim)))
        return np.stack(rows).astype(np.float32)


def effective_weight(seed: SeedSignal) -> float:
    return float(seed.weight * seed.reliability * seed.specificity)


def build_golden_seeds(
    case: GoldenCase,
    category_centroids: dict[str, np.ndarray],
    concept_embeddings: dict[str, np.ndarray],
) -> list[SeedSignal]:
    category_labels: list[str] = []
    concept_keys: list[str] = []
    for tag_name in case.tags:
        if tag_name not in TAG_ALIASES:
            pytest.skip(f"No test alias is defined for tag {tag_name!r}")
        kind, key = TAG_ALIASES[tag_name]
        if kind == "category":
            if not expand_topic_labels([key], TOPIC_LABELS, category_centroids):
                pytest.skip(f"Tag {tag_name!r} has no available category artifact")
            if key not in category_labels:
                category_labels.append(key)
        else:
            if key not in concept_embeddings or key not in CONCEPT_TAG_MAP:
                pytest.skip(f"Tag {tag_name!r} has no available concept artifact")
            if key not in concept_keys:
                concept_keys.append(key)

    seeds: list[SeedSignal] = []
    for code in expand_topic_labels(category_labels, TOPIC_LABELS, category_centroids):
        label = next(
            label for label in category_labels if code in TOPIC_LABELS.get(label, [])
        )
        seeds.append(make_category_seed(code, label, category_centroids[code]))
    for key in concept_keys:
        concept = CONCEPT_TAG_MAP[key]
        seeds.append(
            make_concept_seed(
                key,
                concept.label,
                concept_embeddings[key],
                broad=key in BROAD_CONCEPT_KEYS,
            )
        )
    if case.free_texts:
        model = ArtifactBackedFreeTextModel(concept_embeddings, category_centroids)
        for phrase, embedding in embed_free_text_interests(list(case.free_texts), model):
            seeds.append(make_freetext_seed(phrase, embedding))
    if not seeds:
        pytest.skip(f"Case {case.name!r} produced no usable seeds")
    return seeds


def print_debug_case(case: GoldenCase, seeds: list[SeedSignal], result) -> None:
    debug = result.debug or {}
    print(f"\nCASE {case.name}")
    print(f"expected_k_range={case.expected_k_range} inferred_k={result.centroids.shape[0]}")
    print("idx | label | source | split_power | effective_weight | thread")
    for idx, seed in enumerate(seeds):
        print(
            f"{idx:>3} | {seed.label} | {seed.source} | {seed.split_power:.3f} | "
            f"{effective_weight(seed):.4f} | {int(result.seed_labels[idx])}"
        )
    core_labels = debug.get("core_seed_labels", [])
    support_labels = debug.get("support_seed_labels", [])
    print(f"core_seeds={core_labels}")
    print(f"support_seeds={support_labels}")
    print(f"thread_weights={np.asarray(result.thread_weights).round(4).tolist()}")


@pytest.mark.diagnostic
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case.name for case in GOLDEN_CASES])
def test_golden_onboarding_thread_counts(
    artifact_context: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    case: GoldenCase,
) -> None:
    category_centroids, concept_embeddings = artifact_context
    seeds = build_golden_seeds(case, category_centroids, concept_embeddings)
    result = initialize_user_centroids_threshold(
        seeds,
        max_threads=MAX_THREADS,
        merge_threshold=MERGE_THRESHOLD,
        core_split_power=CORE_SPLIT_POWER,
        debug=True,
    )

    print_debug_case(case, seeds, result)

    lo, hi = case.expected_k_range
    assert lo <= result.centroids.shape[0] <= hi
    assert len(result.seed_labels) == len(seeds)
    assert np.allclose(np.linalg.norm(result.centroids, axis=1), 1.0, atol=1e-4)
    assert np.isclose(result.thread_weights.sum(), 1.0)


def _case_by_name(names: Iterable[str]) -> list[GoldenCase]:
    by_name = {case.name: case for case in GOLDEN_CASES}
    return [by_name[name] for name in names]


@pytest.mark.diagnostic
def test_default_threshold_does_not_totally_collapse_multithread_cases(
    artifact_context: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> None:
    category_centroids, concept_embeddings = artifact_context
    cases = _case_by_name(
        [
            "healthcare_robotics",
            "bio_climate",
            "healthcare_robotics_climate",
            "quantum_healthcare",
        ]
    )

    inferred: dict[str, int] = {}
    for case in cases:
        seeds = build_golden_seeds(case, category_centroids, concept_embeddings)
        result = initialize_user_centroids_threshold(
            seeds,
            max_threads=MAX_THREADS,
            merge_threshold=MERGE_THRESHOLD,
            core_split_power=CORE_SPLIT_POWER,
            debug=True,
        )
        inferred[case.name] = int(result.centroids.shape[0])
        print_debug_case(case, seeds, result)

    assert sum(k_u >= 2 for k_u in inferred.values()) >= 2, inferred
