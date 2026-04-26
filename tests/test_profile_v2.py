"""Tests for the v2 user profile initialization (threshold-based grouping)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.concept_tags import (
    load_concept_embedding_artifacts,
    save_concept_embedding_artifacts,
)
from user.profile import (
    ProfileInitializationResult,
    SeedSignal,
    _effective_weight,
    init_user_profile,
    init_user_profile_v2,
    make_category_seed,
    make_concept_seed,
    make_freetext_seed,
    make_scholar_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    return (v / np.linalg.norm(v)).astype(np.float32)


def _random_unit(dim: int = 768, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.float32)
    return _unit(v)


def _make_seed(
    label: str = "test",
    vector: np.ndarray | None = None,
    weight: float = 1.0,
    reliability: float = 0.9,
    specificity: float = 0.7,
    split_power: float = 0.8,
    source: str = "predefined_tag",
    rng: np.random.Generator | None = None,
) -> SeedSignal:
    if vector is None:
        vector = _random_unit(rng=rng)
    return SeedSignal(
        vector=vector,
        weight=weight,
        reliability=reliability,
        specificity=specificity,
        split_power=split_power,
        label=label,
        source=source,
    )


# ---------------------------------------------------------------------------
# Seed factory tests
# ---------------------------------------------------------------------------

class TestSeedFactories:
    def test_make_category_seed(self):
        emb = _random_unit()
        s = make_category_seed("cs.LG", "Machine Learning", emb)
        assert s.source == "arxiv_category"
        assert s.split_power < 0.6  # support seed
        assert np.allclose(np.linalg.norm(s.vector), 1.0)

    def test_make_concept_seed_specific(self):
        emb = _random_unit()
        s = make_concept_seed("healthcare_ai", "Healthcare AI", emb, broad=False)
        assert s.source == "predefined_tag"
        assert s.split_power >= 0.6  # core seed
        assert s.weight == 1.5

    def test_make_concept_seed_broad(self):
        emb = _random_unit()
        s = make_concept_seed("ml", "ML", emb, broad=True)
        assert s.split_power < 0.6  # support
        assert s.weight == 1.0

    def test_make_scholar_seed(self):
        emb = _random_unit()
        s = make_scholar_seed("A great paper", emb)
        assert s.source == "scholar_title"
        assert s.split_power >= 0.6

    def test_make_freetext_seed(self):
        emb = _random_unit()
        s = make_freetext_seed("diffusion for imaging", emb)
        assert s.source == "free_text"
        assert s.split_power >= 0.6

    def test_effective_weight(self):
        s = _make_seed(weight=2.0, reliability=0.8, specificity=0.5)
        assert abs(_effective_weight(s) - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# init_user_profile_v2 tests
# ---------------------------------------------------------------------------

class TestInitV2:
    def test_empty_seeds_raises(self):
        with pytest.raises(ValueError, match="zero seeds"):
            init_user_profile_v2([])

    def test_single_seed(self):
        s = _make_seed(label="only")
        result = init_user_profile_v2([s])
        assert isinstance(result, ProfileInitializationResult)
        assert result.centroids.shape == (1, 768)
        assert result.thread_labels == ["only"]
        assert result.seed_labels.shape == (1,)
        assert result.seed_labels.tolist() == [0]
        assert np.isclose(result.thread_weights.sum(), 1.0)
        assert np.allclose(np.linalg.norm(result.centroids, axis=1), 1.0)

    def test_two_close_seeds_merge(self):
        """Two nearly identical seeds should merge into a single thread."""
        rng = np.random.default_rng(0)
        base = _random_unit(rng=rng)
        # Perturb slightly — scale 0.005 keeps cosine distance well under 0.22
        noise = rng.standard_normal(768).astype(np.float32) * 0.005
        close = _unit(base + noise)

        s1 = _make_seed(label="A", vector=base, split_power=0.8)
        s2 = _make_seed(label="B", vector=close, split_power=0.8)

        result = init_user_profile_v2([s1, s2])
        assert result.centroids.shape[0] == 1, "Close seeds should merge into 1 thread"
        assert result.seed_labels.shape == (2,)
        assert result.seed_labels.tolist() == [0, 0]
        assert np.isclose(result.thread_weights.sum(), 1.0)

    def test_two_distant_seeds_separate(self):
        """Two orthogonal seeds should become 2 threads."""
        rng = np.random.default_rng(1)
        v1 = np.zeros(768, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(768, dtype=np.float32)
        v2[1] = 1.0

        s1 = _make_seed(label="X", vector=v1, split_power=0.8)
        s2 = _make_seed(label="Y", vector=v2, split_power=0.8)

        result = init_user_profile_v2([s1, s2])
        assert result.centroids.shape[0] == 2, "Distant seeds should remain separate"
        assert len(result.thread_labels) == 2
        assert result.seed_labels.shape == (2,)
        assert len(set(result.seed_labels.tolist())) == 2
        assert np.isclose(result.thread_weights.sum(), 1.0)

    def test_max_threads_cap(self):
        """5 well-separated core seeds should be capped at max_threads=3."""
        seeds = []
        for i in range(5):
            v = np.zeros(768, dtype=np.float32)
            v[i] = 1.0
            seeds.append(_make_seed(label=f"dim{i}", vector=v, split_power=0.9))

        result = init_user_profile_v2(seeds, max_threads=3)
        assert result.centroids.shape[0] <= 3
        assert len(result.thread_labels) == result.centroids.shape[0]
        assert result.seed_labels.shape == (len(seeds),)
        assert result.seed_labels.min() >= 0
        assert result.seed_labels.max() < result.centroids.shape[0]
        assert np.isclose(result.thread_weights.sum(), 1.0)

    def test_support_seed_does_not_form_thread(self):
        """A support seed (low split_power) should attach to the nearest core,
        not create its own thread."""
        v_core = np.zeros(768, dtype=np.float32)
        v_core[0] = 1.0
        v_support = np.zeros(768, dtype=np.float32)
        v_support[0] = 0.9
        v_support[1] = 0.1
        v_support = _unit(v_support)

        core = _make_seed(label="core", vector=v_core, split_power=0.9)
        support = _make_seed(label="support", vector=v_support, split_power=0.2)

        result = init_user_profile_v2([core, support])
        assert result.centroids.shape[0] == 1, "Support seed should merge into core thread"
        assert result.seed_labels.tolist() == [0, 0]

    def test_all_centroids_unit_norm(self):
        rng = np.random.default_rng(7)
        seeds = [_make_seed(label=f"s{i}", rng=rng, split_power=0.8) for i in range(4)]
        result = init_user_profile_v2(seeds)
        norms = np.linalg.norm(result.centroids, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_thread_weights_sum_to_one(self):
        rng = np.random.default_rng(8)
        seeds = [_make_seed(label=f"s{i}", rng=rng, split_power=0.8) for i in range(3)]
        result = init_user_profile_v2(seeds)
        assert np.isclose(result.thread_weights.sum(), 1.0)

    def test_fallback_all_support(self):
        """When all seeds have low split_power, they should all be promoted
        to core and the algorithm should still work."""
        rng = np.random.default_rng(9)
        seeds = [_make_seed(label=f"s{i}", rng=rng, split_power=0.1) for i in range(3)]
        result = init_user_profile_v2(seeds)
        assert result.centroids.shape[0] >= 1
        assert result.seed_labels.shape == (len(seeds),)
        assert np.isclose(result.thread_weights.sum(), 1.0)

    def test_thread_label_from_highest_specificity(self):
        """The thread label should come from the seed with highest specificity."""
        v = _random_unit(rng=np.random.default_rng(10))
        s_low = _make_seed(label="broad", vector=v, specificity=0.3, split_power=0.8)
        # Very small perturbation so both seeds merge into one thread
        noise = np.random.default_rng(11).standard_normal(768).astype(np.float32) * 0.003
        s_high = _make_seed(
            label="specific",
            vector=_unit(v + noise),
            specificity=0.9,
            split_power=0.8,
        )

        result = init_user_profile_v2([s_low, s_high])
        assert result.thread_labels[0] == "specific"


# ---------------------------------------------------------------------------
# Legacy wrapper tests
# ---------------------------------------------------------------------------

class TestLegacyWrapper:
    def test_init_user_profile_returns_correct_shape(self):
        rng = np.random.default_rng(20)
        cat_centroids = {
            "cs.LG": _random_unit(rng=rng),
            "cs.CV": _random_unit(rng=rng),
            "cs.CL": _random_unit(rng=rng),
        }
        centroids = init_user_profile(["cs.LG", "cs.CV"], cat_centroids)
        assert centroids.ndim == 2
        assert centroids.shape[1] == 768
        assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)

    def test_init_user_profile_with_paper_embeddings(self):
        rng = np.random.default_rng(21)
        cat_centroids = {"cs.LG": _random_unit(rng=rng)}
        papers = np.stack([_random_unit(rng=rng) for _ in range(3)])
        centroids = init_user_profile(["cs.LG"], cat_centroids, paper_embeddings=papers)
        assert centroids.ndim == 2
        assert centroids.shape[1] == 768

    def test_init_user_profile_fallback_empty_topics(self):
        cat_centroids = {"cs.LG": _random_unit()}
        centroids = init_user_profile([], cat_centroids)
        assert centroids.shape == (1, 768)


# ---------------------------------------------------------------------------
# Concept embedding artifact tests
# ---------------------------------------------------------------------------

class TestConceptEmbeddingArtifacts:
    def test_save_and_load_round_trip(self, tmp_path):
        embeddings = {
            "healthcare_ai": _unit(np.eye(1, 768, 0, dtype=np.float32).reshape(-1)),
            "medical_imaging": _unit(np.eye(1, 768, 1, dtype=np.float32).reshape(-1)),
        }

        save_concept_embedding_artifacts(embeddings, tmp_path)
        loaded = load_concept_embedding_artifacts(tmp_path)

        assert list(loaded) == ["healthcare_ai", "medical_imaging"]
        assert all(v.dtype == np.float32 for v in loaded.values())
        assert np.allclose(loaded["healthcare_ai"], embeddings["healthcare_ai"])
        assert np.allclose(loaded["medical_imaging"], embeddings["medical_imaging"])

    def test_load_missing_artifacts_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing concept embedding"):
            load_concept_embedding_artifacts(tmp_path)


# ---------------------------------------------------------------------------
# DB round-trip for thread metadata
# ---------------------------------------------------------------------------

class TestDBThreadMetadata:
    def test_create_and_get_with_thread_fields(self, tmp_path):
        from user.db import create_user, get_user, init_db

        db = str(tmp_path / "test.db")
        init_db(db)

        rng = np.random.default_rng(30)
        centroids = np.stack([_random_unit(rng=rng) for _ in range(2)])
        tw = np.array([0.6, 0.4], dtype=np.float32)
        tl = ["Healthcare AI", "Robotics"]

        uid = create_user("Tester", centroids, k_u=2, diversity=0.5,
                          thread_weights=tw, thread_labels=tl)
        user = get_user(uid)

        assert user is not None
        assert user["thread_labels"] == tl
        assert np.allclose(user["thread_weights"], tw)
        assert user["centroids"].shape == (2, 768)

    def test_create_without_thread_fields(self, tmp_path):
        from user.db import create_user, get_user, init_db

        db = str(tmp_path / "test.db")
        init_db(db)

        rng = np.random.default_rng(31)
        centroids = np.stack([_random_unit(rng=rng)])

        uid = create_user("Legacy", centroids, k_u=1, diversity=0.5)
        user = get_user(uid)

        assert user is not None
        assert user["thread_labels"] == ["Thread 1"]
        assert np.allclose(user["thread_weights"], np.array([1.0], dtype=np.float32))

    def test_malformed_thread_fields_fall_back(self, tmp_path):
        import json
        import sqlite3

        from user.db import create_user, get_user, init_db

        db = str(tmp_path / "test.db")
        init_db(db)

        rng = np.random.default_rng(32)
        centroids = np.stack([_random_unit(rng=rng) for _ in range(2)])

        uid = create_user("Malformed", centroids, k_u=2, diversity=0.5)
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE users SET thread_weights = ?, thread_labels = ? WHERE user_id = ?",
            (
                np.array([1.0], dtype=np.float32).tobytes(),
                json.dumps(["too few"]),
                uid,
            ),
        )
        conn.commit()
        conn.close()

        user = get_user(uid)

        assert user is not None
        assert user["thread_labels"] == ["Thread 1", "Thread 2"]
        assert np.allclose(user["thread_weights"], np.array([0.5, 0.5], dtype=np.float32))

    def test_init_db_idempotent(self, tmp_path):
        from user.db import init_db

        db = str(tmp_path / "test.db")
        init_db(db)
        init_db(db)  # second call should not error
