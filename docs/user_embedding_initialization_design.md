# User Embedding Initialization Design

## ArXiv Recommendation Engine

This document records the revised design for improving user embedding initialization in the ArXiv recommendation system.

The goal is to make cold-start user profiles more precise, especially for interdisciplinary interests such as **machine learning + healthcare**, while keeping onboarding lightweight and computationally efficient.

The current algorithm initializes a user profile from selected arXiv category centroids, optionally enriched with Google Scholar signals. These seed vectors are clustered into up to three user centroids. This is a good high-level direction, but the current approach has two major weaknesses:

1. arXiv categories are not expressive enough as the only onboarding vocabulary.
2. k-means clustering is not ideal for small seed sets and requires a predefined number of clusters.

This revision addresses both problems.

---

# Problem 1: More Precise Onboarding Signals

## 1.1 Problem Statement

The current category-based onboarding has several limitations.

First, some user-facing interests do not correspond cleanly to arXiv categories. For example, there is no single arXiv category for:

- Healthcare AI
- Climate & Weather
- Computational Biology, in the broad user-facing sense
- Scientific Machine Learning
- AI for Education
- Finance & Economics AI

Second, arXiv categories are not always intuitive for users. A user may understand “Healthcare AI” or “ML for Biology,” but not know whether to select `cs.LG`, `cs.CV`, `cs.CL`, `stat.ML`, `q-bio.QM`, or some combination of them.

Third, highly specific research directions may be necessary for good initialization, but they are too narrow to list as fixed onboarding choices. For example:

- single-cell perturbation modeling
- diffusion models for medical imaging
- LLMs for clinical decision support
- neural operators for weather forecasting

A UI with hundreds of such options would be overwhelming.

Therefore, the onboarding vocabulary should be separated from the embedding-construction vocabulary.

The user-facing interface should stay simple and readable, while the embedding backend should use richer hidden descriptions and seed vectors.

---

## 1.2 Adopted Solution A: User Free-Text Interests

Users should be allowed to enter optional free-text interests during onboarding.

Examples:

```text
machine learning for healthcare
single-cell perturbation modeling
diffusion models for medical imaging
LLMs for clinical decision support
climate forecasting with neural operators
computational neuroscience and representation learning
```

These free-text interests become additional seed signals for user embedding initialization.

### Important Detail: Do Not Embed Raw Short Phrases Directly

A phrase such as:

```text
healthcare
```

is too short and underspecified. Since the paper embedding model is trained on scientific text, not isolated casual tags, short user text should be expanded into a richer pseudo-query before embedding.

Example:

```python
def expand_free_text_interest(text: str) -> str:
    return (
        f"Research papers about {text}. "
        f"This topic includes relevant methods, applications, datasets, "
        f"benchmarks, and recent scientific progress."
    )
```

For example:

```text
Input:
    healthcare

Expanded embedding text:
    Research papers about healthcare. This topic includes relevant methods,
    applications, datasets, benchmarks, and recent scientific progress.
```

For better results, important known concepts can use curated expansions:

```python
FREE_TEXT_EXPANSIONS = {
    "healthcare": (
        "machine learning for healthcare, clinical prediction, electronic health "
        "records, medical imaging, biomedical data, diagnosis, treatment planning, "
        "and clinical decision support"
    ),
    "computational biology": (
        "computational biology, genomics, single-cell data, protein modeling, "
        "biological networks, perturbation modeling, and bioinformatics"
    ),
    "climate": (
        "climate modeling, weather forecasting, meteorological downscaling, "
        "earth system science, and climate data analysis"
    ),
}


def expand_free_text_interest(text: str) -> str:
    normalized = text.strip().lower()

    if normalized in FREE_TEXT_EXPANSIONS:
        expanded = FREE_TEXT_EXPANSIONS[normalized]
    else:
        expanded = text

    return (
        f"Research papers about {expanded}. "
        f"This topic includes relevant methods, applications, datasets, "
        f"benchmarks, and recent scientific progress."
    )
```

---

## 1.3 Adopted Solution B: Predefined Non-arXiv Concept Tags

In addition to arXiv categories, the system should support a curated registry of human-readable concept tags.

These tags are not required to correspond one-to-one with arXiv categories.

Examples:

```text
Healthcare AI
Climate & Weather
Computational Biology
Scientific Machine Learning
Computational Neuroscience
AI for Education
Finance & Economics AI
Medical Imaging
LLMs
Robotics
Human-Computer Interaction
Security & Privacy
```

Each concept tag should have:

- a human-readable label,
- a short UI description,
- richer hidden seed texts,
- optional related arXiv categories,
- a precomputed embedding.

### Concept Tag Data Structure

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class ConceptTag:
    key: str
    label: str
    description: str
    seed_texts: list[str]
    related_arxiv_categories: list[str]
    embedding: np.ndarray | None = None
```

### Example Concept Tags

```python
CONCEPT_TAGS = {
    "healthcare_ai": ConceptTag(
        key="healthcare_ai",
        label="Healthcare AI",
        description="Machine learning for clinical, biomedical, and healthcare applications.",
        seed_texts=[
            "machine learning for healthcare and clinical decision support",
            "deep learning for medical imaging and diagnosis",
            "predictive modeling with electronic health records",
            "biomedical natural language processing",
            "AI systems for diagnosis, treatment planning, and patient data analysis",
        ],
        related_arxiv_categories=["cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML", "q-bio.QM"],
    ),

    "computational_biology": ConceptTag(
        key="computational_biology",
        label="Computational Biology",
        description="Computational methods for biological data, genomics, proteins, and cellular systems.",
        seed_texts=[
            "computational biology and bioinformatics machine learning",
            "single-cell genomics and perturbation modeling",
            "protein structure prediction and biological sequence modeling",
            "machine learning for biological networks and cellular systems",
        ],
        related_arxiv_categories=["q-bio.QM", "q-bio.GN", "q-bio.MN", "cs.LG", "stat.ML"],
    ),

    "climate_weather": ConceptTag(
        key="climate_weather",
        label="Climate & Weather",
        description="Modeling, forecasting, and analyzing climate, weather, and earth-system data.",
        seed_texts=[
            "machine learning for weather forecasting and climate modeling",
            "meteorological downscaling and bias correction",
            "earth system science and climate data analysis",
            "neural operators and deep learning for physical forecasting systems",
        ],
        related_arxiv_categories=["physics.ao-ph", "cs.LG", "stat.ML", "math.NA"],
    ),

    "scientific_ml": ConceptTag(
        key="scientific_ml",
        label="Scientific Machine Learning",
        description="Machine learning for scientific simulation, physical systems, and data-driven discovery.",
        seed_texts=[
            "scientific machine learning for physical systems",
            "neural operators for partial differential equations",
            "machine learning for scientific simulation and discovery",
            "physics-informed machine learning and surrogate modeling",
        ],
        related_arxiv_categories=["cs.LG", "physics.comp-ph", "math.NA", "stat.ML"],
    ),
}
```

---

## 1.4 Precomputing Concept Tag Embeddings

Each concept tag should have a precomputed embedding. This can be computed from the tag's hidden seed texts.

```python
def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def compute_concept_embedding(concept: ConceptTag, embed_fn) -> np.ndarray:
    texts = [
        f"Research papers about {text}."
        for text in concept.seed_texts
    ]

    embs = embed_fn(texts)              # shape: (n_texts, 768)
    embs = normalize_rows(embs)

    return normalize(embs.mean(axis=0))
```



---

# Problem 2: Consistent, Effective, and Efficient User Embedding Construction

## 2.1 Problem Statement

After collecting onboarding seed signals, the system needs to convert them into one or more user centroids.

The current approach uses k-means over seed vectors. However, this has several problems:

1. The number of clusters `k_u` must be chosen before clustering.
2. Small seed sets are common during onboarding.
3. K-means on 2 to 5 points can behave arbitrarily.
4. The system should create multiple threads only when seed vectors are meaningfully far apart.
5. The number of user threads should be bounded, usually by `max_threads = 3`.

Therefore, we replace k-means as the default initializer with:

```text
Threshold-based thread inference
```

The central idea is:

```text
Keep seed signals together if they are close in embedding space.
Separate them into different threads only when their cosine distance exceeds a threshold.
```

---

## 2.2 Core Geometry

All embeddings are unit-normalized.

Similarity:

```python
similarity(u, v) = u @ v
```

Cosine distance:

```python
distance(u, v) = 1 - similarity(u, v)
```

Interpretation:

```text
small distance  -> semantically close
large distance  -> semantically far
```

Example rough ranges, to be tuned empirically:

```text
distance < 0.15       very close
0.15 to 0.25          related but somewhat different
distance > 0.25       likely different research threads
```

Initial recommended threshold:

```python
MERGE_THRESHOLD = 0.22
MAX_THREADS = 3
```

The exact threshold should be tuned by inspecting real pairwise distances among concept tags and free-text examples.

---

## 2.3 Seed Signal Structure

Each onboarding signal becomes a `SeedSignal`.

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class SeedSignal:
    text: str
    vector: np.ndarray
    source: Literal[
        "predefined_tag",
        "free_text",
        "arxiv_category",
        "scholar_title",
        "example_paper",
    ]
    weight: float
    reliability: float
    specificity: float
    split_power: float
    label: str | None = None
```

The final effective centroid weight is:

```python
effective_weight = weight * reliability * specificity
```

The `split_power` controls whether a seed is allowed to create a new research thread.

This distinction is important.

A context-only category seed such as:

```text
Machine Learning
```

can contribute to the final centroid while using lower split power than a specific concept or free-text seed.

A specific free-text interest such as:

```text
diffusion models for medical imaging
```

should have stronger ability to create a distinct thread.

Example source settings:

```python
SEED_SOURCE_CONFIG = {
    "predefined_tag": {
        "weight": 1.5,
        "reliability": 0.9,
        "specificity": 0.7,
        "split_power": 0.7,
    },
    "free_text": {
        "weight": 2.0,
        "reliability": 0.8,
        "specificity": 0.9,
        "split_power": 1.0,
    },
}
```

---

# 3. Agglomerative Threshold Grouping

## 3.1 Algorithm Intuition

Agglomerative threshold grouping starts with each seed as its own group.

Then it repeatedly merges the two closest groups if their distance is below a threshold.

It stops when all remaining groups are farther apart than the threshold.

Finally, if there are still more than `max_threads` groups, it merges the closest remaining groups until the number of groups is bounded.

This directly matches the desired behavior:

```text
Do not predefine the number of threads.
Infer the number of threads from embedding geometry.
Allow multiple threads only when seed groups are actually separated.
Bound the final number of threads.
```

---

## 3.2 Group Centroid

Each group has a weighted centroid:

```python
centroid(group) = normalize(sum_i weight_i * vector_i)
```

where the sum is over seeds in that group.

```python
def weighted_centroid(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    X = np.stack(vectors)
    w = np.array(weights)
    return normalize((X * w[:, None]).sum(axis=0))
```

---

## 3.3 Group Distance

The distance between two groups is the cosine distance between their weighted centroids:

```python
distance(group_a, group_b) = 1 - centroid(group_a) @ centroid(group_b)
```

```python
def group_distance(group_a, group_b) -> float:
    ca = group_centroid(group_a)
    cb = group_centroid(group_b)
    return 1.0 - float(ca @ cb)
```

---

## 3.4 Agglomerative Threshold Grouping Code

```python
import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def effective_seed_weight(seed: SeedSignal) -> float:
    return seed.weight * seed.reliability * seed.specificity


def threshold_agglomerative_grouping(
    seeds: list[SeedSignal],
    merge_threshold: float = 0.22,
    max_threads: int = 3,
):
    """
    Groups seed vectors into user-interest threads.

    The number of groups is inferred from cosine-distance geometry and then
    bounded by max_threads.

    Args:
        seeds:
            List of seed signals. Each seed.vector must be unit-normalized.
        merge_threshold:
            If the distance between two groups is <= merge_threshold, they are
            considered close enough to belong to the same thread.
        max_threads:
            Maximum number of user interest threads.

    Returns:
        centroids:
            np.ndarray of shape (k_u, 768), unit-normalized.
        labels:
            np.ndarray of shape (n_seeds,), assigning each seed to a thread.
        thread_weights:
            np.ndarray of shape (k_u,), normalized to sum to 1.
    """

    if len(seeds) == 0:
        raise ValueError("Cannot initialize user profile with zero seeds.")

    if len(seeds) == 1:
        return (
            seeds[0].vector.reshape(1, -1),
            np.array([0], dtype=int),
            np.array([1.0]),
        )

    groups = [
        {
            "indices": [i],
            "vectors": [seed.vector],
            "weights": [effective_seed_weight(seed)],
        }
        for i, seed in enumerate(seeds)
    ]

    def group_centroid(group):
        X = np.stack(group["vectors"])
        w = np.array(group["weights"])
        return normalize((X * w[:, None]).sum(axis=0))

    def group_distance(g1, g2):
        c1 = group_centroid(g1)
        c2 = group_centroid(g2)
        return 1.0 - float(c1 @ c2)

    # Phase 1: merge groups that are close enough.
    while True:
        best_pair = None
        best_dist = float("inf")

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dist = group_distance(groups[i], groups[j])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        if best_pair is None or best_dist > merge_threshold:
            break

        i, j = best_pair

        groups[i]["indices"].extend(groups[j]["indices"])
        groups[i]["vectors"].extend(groups[j]["vectors"])
        groups[i]["weights"].extend(groups[j]["weights"])

        del groups[j]

    # Phase 2: enforce max_threads by merging closest remaining groups.
    while len(groups) > max_threads:
        best_pair = None
        best_dist = float("inf")

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dist = group_distance(groups[i], groups[j])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair

        groups[i]["indices"].extend(groups[j]["indices"])
        groups[i]["vectors"].extend(groups[j]["vectors"])
        groups[i]["weights"].extend(groups[j]["weights"])

        del groups[j]

    # Final centroids and labels.
    centroids = []
    thread_weights = []
    labels = np.empty(len(seeds), dtype=int)

    for group_id, group in enumerate(groups):
        c = group_centroid(group)
        centroids.append(c)

        total_weight = sum(group["weights"])
        thread_weights.append(total_weight)

        for idx in group["indices"]:
            labels[idx] = group_id

    centroids = normalize_rows(np.stack(centroids))

    thread_weights = np.array(thread_weights)
    thread_weights = thread_weights / thread_weights.sum()

    return centroids, labels, thread_weights
```

---

# 4. Final Version: Threshold-Based Thread Inference

The final recommended initializer refines agglomerative grouping by distinguishing between:

```text
core seeds
support seeds
```

## 4.1 Core Seeds

Core seeds are allowed to create research threads.

Examples:

```text
free-text interests
specific predefined tags
example papers, if later added
```

These usually have high `split_power`.

## 4.2 Support Seeds

Support seeds help shape the final centroid, but should not easily create their own thread.

Examples include arXiv category seeds and any other context-only seeds.
These usually have low `split_power`.

---

## 4.3 Final Algorithm

```text
Input:
    seed signals from predefined concept tags and free-text interests

Each seed has:
    vector
    weight
    reliability
    specificity
    split_power

Step 1:
    Separate seeds into core seeds and support seeds.

Step 2:
    Run agglomerative threshold grouping on core seeds.

Step 3:
    Enforce max_threads.

Step 4:
    Assign all support seeds to the nearest inferred core thread.

Step 5:
    Recompute final weighted normalized centroids using all seeds.

Step 6:
    Return centroids, seed labels, and thread weights.
```

---

## 4.4 Final Initializer Code

```python
def initialize_user_centroids_threshold(
    seeds: list[SeedSignal],
    max_threads: int = 3,
    merge_threshold: float = 0.22,
    core_split_power: float = 0.6,
):
    """
    Initialize user interest centroids from onboarding seed signals.

    This is the recommended replacement for small-seed k-means initialization.

    The algorithm:
        1. separates thread-forming core seeds from support seeds;
        2. infers thread groups using agglomerative threshold grouping;
        3. assigns all seeds to the inferred threads;
        4. computes final weighted centroids.

    Args:
        seeds:
            List of SeedSignal objects.
        max_threads:
            Maximum number of user centroids.
        merge_threshold:
            Maximum cosine distance for two groups to be merged.
        core_split_power:
            Minimum split_power required for a seed to create a thread.

    Returns:
        centroids:
            np.ndarray of shape (k_u, 768), unit-normalized.
        labels:
            np.ndarray of shape (n_seeds,), assigning each seed to a thread.
        thread_weights:
            np.ndarray of shape (k_u,), normalized to sum to 1.
    """

    if len(seeds) == 0:
        raise ValueError("Cannot initialize user profile with zero seeds.")

    if len(seeds) == 1:
        return (
            seeds[0].vector.reshape(1, -1),
            np.array([0], dtype=int),
            np.array([1.0]),
        )

    # 1. Separate thread-forming seeds from support seeds.
    core_seeds = [s for s in seeds if s.split_power >= core_split_power]
    support_seeds = [s for s in seeds if s.split_power < core_split_power]

    # Fallback: if no seed is specific enough, use all seeds as core.
    if len(core_seeds) == 0:
        core_seeds = seeds
        support_seeds = []

    # 2. Infer thread groups using only core seeds.
    core_centroids, _, _ = threshold_agglomerative_grouping(
        core_seeds,
        merge_threshold=merge_threshold,
        max_threads=max_threads,
    )

    # 3. Assign all seeds to the nearest inferred core centroid.
    all_vectors = np.stack([s.vector for s in seeds])
    sims = all_vectors @ core_centroids.T
    labels = np.argmax(sims, axis=1)

    # 4. Recompute final weighted centroids using all seeds.
    weights = np.array([effective_seed_weight(s) for s in seeds])

    centroids = []
    thread_weights = []

    for j in range(len(core_centroids)):
        idx = np.where(labels == j)[0]

        if len(idx) == 0:
            continue

        Xj = all_vectors[idx]
        wj = weights[idx]

        cj = normalize((Xj * wj[:, None]).sum(axis=0))
        centroids.append(cj)
        thread_weights.append(wj.sum())

    centroids = normalize_rows(np.stack(centroids))

    thread_weights = np.array(thread_weights)
    thread_weights = thread_weights / thread_weights.sum()

    return centroids, labels, thread_weights
```

---

# 5. Building Seed Signals from Onboarding Inputs

## 5.1 From Predefined Concept Tags

```python
def seed_from_concept_tag(concept: ConceptTag) -> SeedSignal:
    if concept.embedding is None:
        raise ValueError(f"Concept {concept.key} has no precomputed embedding.")

    return SeedSignal(
        text=concept.description,
        vector=concept.embedding,
        source="predefined_tag",
        weight=1.5,
        reliability=0.9,
        specificity=0.7,
        split_power=0.7,
        label=concept.label,
    )
```

---

## 5.2 From User Free Text

```python
def seed_from_free_text(text: str, embed_fn) -> SeedSignal:
    expanded_text = expand_free_text_interest(text)
    vector = normalize(embed_fn([expanded_text])[0])

    return SeedSignal(
        text=expanded_text,
        vector=vector,
        source="free_text",
        weight=2.0,
        reliability=0.8,
        specificity=0.9,
        split_power=1.0,
        label=text,
    )
```

---

## 5.3 Full Seed Construction

```python
def build_onboarding_seeds(
    selected_concept_keys: list[str],
    free_text_interests: list[str],
    concept_registry: dict[str, ConceptTag],
    embed_fn,
) -> list[SeedSignal]:
    seeds: list[SeedSignal] = []

    for key in selected_concept_keys:
        concept = concept_registry[key]
        seeds.append(seed_from_concept_tag(concept))

    for text in free_text_interests:
        if text.strip():
            seeds.append(seed_from_free_text(text, embed_fn))

    return seeds
```

---



# 7. Small-Seed Behavior

The initializer should handle small seed sets explicitly.

```text
n_seeds = 1:
    Return one centroid.

n_seeds = 2:
    Return one or two centroids depending on distance and split_power.

3 <= n_seeds:
    Use threshold-based thread inference.
```

The implemented algorithm already handles `n_seeds = 1` directly.

For `n_seeds = 2`, agglomerative grouping naturally does the right thing:

- If the two seeds are close, they merge.
- If they are far, they remain separate.

This is preferable to k-means, which would create two clusters whenever `k=2` regardless of whether the two seeds are truly distinct.

---

# 8. Example Behaviors

## Example 1: Overlapping ML Tags

Input:

```text
Machine Learning
Deep Learning
Neural Networks
```

Expected behavior:

```text
One thread:
    Machine Learning / Deep Learning / Neural Networks
```

Reason:

The tags are semantically close and should not become separate user centroids.

---

## Example 2: ML + Healthcare

Input:

```text
Machine Learning
Healthcare AI
free text: diffusion models for medical imaging
```

Expected behavior:

```text
One or two threads depending on actual embedding distance:

Likely Thread 1:
    Healthcare AI
    diffusion models for medical imaging
    Machine Learning
```

Potentially:

```text
Thread 1:
    Healthcare AI
    diffusion models for medical imaging

Thread 2:
    Machine Learning
```

The second case should happen only if the ML vector is genuinely far enough from the healthcare/free-text signals under the merge threshold.

---

## Example 3: Interdisciplinary User with Two Real Threads

Input:

```text
Healthcare AI
free text: diffusion models for medical imaging
Robotics
free text: reinforcement learning for robot navigation
```

Expected behavior:

```text
Thread 1:
    Healthcare AI
    diffusion models for medical imaging

Thread 2:
    Robotics
    reinforcement learning for robot navigation
```

Reason:

The healthcare/medical-imaging seeds and robotics/RL seeds should be far enough in embedding space to form distinct threads.

---

## Example 4: Broad + Specific Concepts

Input:

```text
Artificial Intelligence
Machine Learning
free text: single-cell perturbation modeling
```

Expected behavior:

```text
Thread 1:
    single-cell perturbation modeling
    Machine Learning as support
    Artificial Intelligence as support
```

Reason:

The specific free-text interest should dominate the actual thread formation. Broad AI/ML tags should contribute context but should not create extra centroids by themselves.

---

# 9. Threshold Tuning Plan

The most important hyperparameter is:

```python
merge_threshold = 0.22
```

This should be tuned empirically.

Recommended diagnostic script:

```python
def inspect_concept_distances(concept_registry: dict[str, ConceptTag]):
    keys = list(concept_registry.keys())
    X = np.stack([concept_registry[k].embedding for k in keys])
    X = normalize_rows(X)

    D = 1.0 - X @ X.T

    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs.append((D[i, j], keys[i], keys[j]))

    pairs.sort()

    print("Closest pairs:")
    for d, a, b in pairs[:20]:
        print(f"{d:.3f}\t{a}\t{b}")

    print("\nFarthest pairs:")
    for d, a, b in pairs[-20:]:
        print(f"{d:.3f}\t{a}\t{b}")
```

Then manually inspect distances for pairs such as:

```text
Machine Learning vs Deep Learning
Machine Learning vs Healthcare AI
Healthcare AI vs Medical Imaging
NLP vs LLMs
Computer Vision vs Medical Imaging
Healthcare AI vs Astrophysics
Robotics vs Number Theory
Climate & Weather vs Scientific Machine Learning
```

The threshold should be chosen so that obviously related concepts merge, while clearly different areas remain separate.

---

# 10. Validation Tests

Before integrating the initializer into the online recommender, test it independently.

Example test cases:

```python
TEST_CASES = [
    {
        "name": "overlapping_ml",
        "concepts": ["machine_learning", "deep_learning", "neural_networks"],
        "free_text": [],
    },
    {
        "name": "ml_healthcare",
        "concepts": ["machine_learning", "healthcare_ai"],
        "free_text": ["diffusion models for medical imaging"],
    },
    {
        "name": "healthcare_robotics",
        "concepts": ["healthcare_ai", "robotics"],
        "free_text": [
            "diffusion models for medical imaging",
            "reinforcement learning for robot navigation",
        ],
    },
    {
        "name": "category_context_plus_specific_bio",
        "concepts": ["artificial_intelligence", "machine_learning"],
        "free_text": ["single-cell perturbation modeling"],
    },
]
```

For each case, print:

```text
1. selected seeds
2. pairwise distance matrix
3. inferred number of threads
4. seed assignment for each thread
5. final thread labels
```

Diagnostic function:

```python
def debug_initialization(profile, seeds, labels):
    print(f"Inferred k_u = {profile['k_u']}")

    X = np.stack([s.vector for s in seeds])
    D = 1.0 - X @ X.T

    print("\nPairwise seed distances:")
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            if i < j:
                print(f"{D[i, j]:.3f}\t{si.label}\t{sj.label}")

    print("\nThread assignments:")
    for j in range(profile["k_u"]):
        print(f"\nThread {j}: {profile['thread_labels'][j]}")
        idx = np.where(labels == j)[0]
        for i in idx:
            print(f"  - {seeds[i].label} [{seeds[i].source}]")
```

---

# 11. Why This Replaces K-Means for Initialization

K-means is still a valid algorithm, but it is not ideal as the default cold-start initializer here.

The onboarding seed set is usually small. The system does not need to discover many hidden clusters from a large dataset. It needs to decide whether a few user-provided signals represent one, two, or three coherent research threads.

Threshold-based thread inference is better suited for this because it is:

```text
consistent
computationally cheap
interpretable
stable for small seed sets
does not require predefined k
bounded by max_threads
aligned with the product meaning of research threads
```

The output remains compatible with the rest of the recommendation system:

```python
UserProfile {
    centroids: np.ndarray        # shape (k_u, 768)
    k_u: int
    thread_weights: np.ndarray  # shape (k_u,)
    thread_labels: list[str]
    diversity: float
}
```

Downstream retrieval can still use:

```python
sims = (E_sub @ centroids.T).max(axis=1)
```

So the serving pipeline does not need major changes.

---

# 12. Summary of Adopted Changes

## Problem 1: More Precise Onboarding Signals

Adopted:

```text
1. User free-text interests as additional seed signals.
2. Predefined non-arXiv concept tags such as Healthcare AI, Climate & Weather, Computational Biology, etc.
```

Not adopted yet:

```text
3. User onboarding profile type, such as explore broadly / learn a field / track my research.
```

This can be added later, but it would require broader changes to ranking, diversity, and feedback behavior.

---

## Problem 2: Consistent, Effective, and Efficient User Embedding Construction

Adopted:

```text
Threshold-based thread inference
```

Core mechanism:

```text
1. Build weighted seed signals.
2. Separate core seeds from support seeds using split_power.
3. Run agglomerative threshold grouping on core seeds.
4. Assign support seeds to the nearest inferred thread.
5. Compute final weighted normalized centroids.
6. Return 1 to max_threads user centroids.
```

This replaces k-means as the default cold-start initializer.

---

# 13. Implementation Checklist

- [ ] Add `ConceptTag` registry.
- [ ] Precompute embeddings for predefined concept tags.
- [ ] Add free-text interest input to onboarding UI.
- [ ] Implement free-text expansion before embedding.
- [ ] Implement `SeedSignal` structure.
- [ ] Implement seed construction from concept tags and free text.
- [ ] Implement `threshold_agglomerative_grouping`.
- [ ] Implement `initialize_user_centroids_threshold`.
- [ ] Store `thread_weights` and `thread_labels` in `UserProfile`.
- [ ] Add debug view for pairwise seed distances and thread assignments.
- [ ] Tune `merge_threshold` using real concept embeddings.
- [ ] Compare against old k-means initialization on synthetic onboarding cases.
