"""Predefined concept tag registry for onboarding.

Concept tags provide human-readable, interdisciplinary research themes that
go beyond raw arXiv category codes. Each tag carries curated seed texts
whose embeddings are averaged to produce a single representative vector.

The registry is a pure-data module with no UI or DB dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class ConceptTag:
    """A curated research theme used during onboarding.

    Attributes:
        key: Unique identifier, e.g. ``"healthcare_ai"``.
        label: Human-readable name shown in the UI.
        description: Short blurb for tooltip / help text.
        seed_texts: Richer descriptions embedded and averaged to form the
            tag's representative vector.  Written as realistic paper-title
            style strings so SPECTER2 produces high-quality embeddings.
        related_arxiv_categories: arXiv codes loosely associated with this
            theme (informational; not used for embedding).
        embedding: Populated by :func:`compute_concept_embeddings` or by
            loading the standalone concept embedding artifact.
    """

    key: str
    label: str
    description: str
    seed_texts: list[str]
    related_arxiv_categories: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONCEPT_TAGS: list[ConceptTag] = [
    ConceptTag(
        key="healthcare_ai",
        label="Healthcare AI",
        description="Machine learning for clinical, biomedical, and healthcare applications.",
        seed_texts=[
            "Machine learning for healthcare and clinical decision support",
            "Deep learning for medical imaging and diagnosis",
            "Predictive modeling with electronic health records",
            "Biomedical natural language processing",
            "AI systems for diagnosis, treatment planning, and patient data analysis",
        ],
        related_arxiv_categories=["cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML", "q-bio.QM"],
    ),
    ConceptTag(
        key="medical_imaging",
        label="Medical Imaging",
        description="Deep learning applied to radiology, pathology, and ophthalmology images.",
        seed_texts=[
            "Deep learning for radiology and pathology image analysis",
            "Medical image segmentation with convolutional neural networks",
            "Computer-aided diagnosis from chest X-rays and CT scans",
            "Diffusion models for medical image synthesis and augmentation",
        ],
        related_arxiv_categories=["cs.CV", "cs.LG", "eess.IV"],
    ),
    ConceptTag(
        key="computational_biology",
        label="Computational Biology",
        description="Computational methods for biological data, genomics, proteins, and cellular systems.",
        seed_texts=[
            "Computational biology and bioinformatics machine learning",
            "Single-cell genomics and perturbation modeling",
            "Protein structure prediction and biological sequence modeling",
            "Machine learning for biological networks and cellular systems",
        ],
        related_arxiv_categories=["q-bio.QM", "q-bio.GN", "q-bio.MN", "cs.LG", "stat.ML"],
    ),
    ConceptTag(
        key="drug_discovery",
        label="Drug Discovery",
        description="AI-driven molecular generation, virtual screening, and drug design.",
        seed_texts=[
            "Machine learning for drug discovery and molecular generation",
            "Graph neural networks for molecular property prediction",
            "Virtual screening and structure-based drug design with deep learning",
            "Generative models for de novo molecule design",
        ],
        related_arxiv_categories=["q-bio.QM", "cs.LG", "physics.chem-ph"],
    ),
    ConceptTag(
        key="climate_weather",
        label="Climate & Weather",
        description="Modeling, forecasting, and analyzing climate, weather, and earth-system data.",
        seed_texts=[
            "Machine learning for weather forecasting and climate modeling",
            "Meteorological downscaling and bias correction",
            "Earth system science and climate data analysis",
            "Neural operators and deep learning for physical forecasting systems",
        ],
        related_arxiv_categories=["physics.ao-ph", "cs.LG", "stat.ML", "math.NA"],
    ),
    ConceptTag(
        key="scientific_ml",
        label="Scientific Machine Learning",
        description="ML for scientific simulation, physical systems, and data-driven discovery.",
        seed_texts=[
            "Scientific machine learning for physical systems",
            "Neural operators for partial differential equations",
            "Machine learning for scientific simulation and discovery",
            "Physics-informed machine learning and surrogate modeling",
        ],
        related_arxiv_categories=["cs.LG", "physics.comp-ph", "math.NA", "stat.ML"],
    ),
    ConceptTag(
        key="llms",
        label="Large Language Models",
        description="Training, alignment, reasoning, and applications of large language models.",
        seed_texts=[
            "Large language model pre-training and scaling laws",
            "Instruction tuning and reinforcement learning from human feedback",
            "Reasoning and chain-of-thought prompting in language models",
            "Retrieval-augmented generation and tool use in LLMs",
        ],
        related_arxiv_categories=["cs.CL", "cs.AI", "cs.LG"],
    ),
    ConceptTag(
        key="generative_models",
        label="Generative Models",
        description="Diffusion models, GANs, VAEs, and flow-based generative methods.",
        seed_texts=[
            "Denoising diffusion probabilistic models for image generation",
            "Score-based generative modeling and stochastic differential equations",
            "Variational autoencoders and normalizing flows",
            "Text-to-image generation and controllable synthesis",
        ],
        related_arxiv_categories=["cs.LG", "cs.CV", "stat.ML"],
    ),
    ConceptTag(
        key="graph_neural_networks",
        label="Graph Neural Networks",
        description="Message-passing, spectral, and transformer-based learning on graphs.",
        seed_texts=[
            "Graph neural networks and message passing on relational data",
            "Graph transformers and positional encodings for graph learning",
            "Link prediction and node classification with GNNs",
            "Scalable graph learning and heterogeneous graph networks",
        ],
        related_arxiv_categories=["cs.LG", "cs.SI", "stat.ML"],
    ),
    ConceptTag(
        key="reinforcement_learning",
        label="Reinforcement Learning",
        description="Policy optimization, multi-agent RL, and decision-making under uncertainty.",
        seed_texts=[
            "Deep reinforcement learning and policy gradient methods",
            "Offline reinforcement learning from logged data",
            "Multi-agent reinforcement learning and cooperative strategies",
            "Model-based reinforcement learning and world models",
        ],
        related_arxiv_categories=["cs.LG", "cs.AI", "cs.MA"],
    ),
    ConceptTag(
        key="ai_safety",
        label="AI Safety & Alignment",
        description="Alignment, interpretability, robustness, and safety of AI systems.",
        seed_texts=[
            "AI alignment and value learning for safe AI systems",
            "Mechanistic interpretability and neural network understanding",
            "Adversarial robustness and certified defenses",
            "Red-teaming and evaluation of large language model safety",
        ],
        related_arxiv_categories=["cs.AI", "cs.LG", "cs.CL"],
    ),
    ConceptTag(
        key="federated_privacy",
        label="Federated & Privacy-Preserving ML",
        description="Federated learning, differential privacy, and secure computation.",
        seed_texts=[
            "Federated learning for distributed model training",
            "Differential privacy guarantees in machine learning",
            "Secure multi-party computation for private inference",
            "Privacy-preserving data analysis and synthetic data generation",
        ],
        related_arxiv_categories=["cs.LG", "cs.CR", "stat.ML"],
    ),
    ConceptTag(
        key="multimodal_learning",
        label="Multimodal Learning",
        description="Vision-language models, audio-visual learning, and cross-modal alignment.",
        seed_texts=[
            "Vision-language models and cross-modal representation learning",
            "Multimodal fusion for image, text, and audio understanding",
            "Contrastive learning for multimodal alignment",
            "Visual question answering and image captioning",
        ],
        related_arxiv_categories=["cs.CV", "cs.CL", "cs.LG", "cs.MM"],
    ),
    ConceptTag(
        key="robotics_embodied",
        label="Robotics & Embodied AI",
        description="Robot learning, manipulation, navigation, and embodied intelligence.",
        seed_texts=[
            "Robot learning for manipulation and grasping",
            "Embodied AI agents in simulated and real environments",
            "Reinforcement learning for robot navigation and locomotion",
            "Foundation models for robotic control and planning",
        ],
        related_arxiv_categories=["cs.RO", "cs.AI", "cs.LG", "cs.CV"],
    ),
    ConceptTag(
        key="autonomous_driving",
        label="Autonomous Driving",
        description="Self-driving perception, planning, and end-to-end driving models.",
        seed_texts=[
            "End-to-end autonomous driving with deep learning",
            "3D object detection and tracking for self-driving vehicles",
            "Motion planning and prediction for autonomous vehicles",
            "Simulation and domain adaptation for autonomous driving",
        ],
        related_arxiv_categories=["cs.CV", "cs.RO", "cs.AI", "cs.LG"],
    ),
    ConceptTag(
        key="computational_neuroscience",
        label="Computational Neuroscience",
        description="Neural coding, brain-inspired models, and neural data analysis.",
        seed_texts=[
            "Computational models of neural coding and brain function",
            "Machine learning for neural data analysis and brain-computer interfaces",
            "Brain-inspired learning algorithms and spiking neural networks",
            "Representation learning and neural population dynamics",
        ],
        related_arxiv_categories=["q-bio.NC", "cs.NE", "cs.LG", "stat.ML"],
    ),
    ConceptTag(
        key="ai_for_code",
        label="AI for Code",
        description="Code generation, program synthesis, and software engineering with AI.",
        seed_texts=[
            "Large language models for code generation and completion",
            "Program synthesis and automated software engineering",
            "Code understanding, summarization, and bug detection",
            "Neural program repair and test generation",
        ],
        related_arxiv_categories=["cs.SE", "cs.CL", "cs.PL", "cs.LG"],
    ),
    ConceptTag(
        key="time_series",
        label="Time Series Forecasting",
        description="Deep learning for temporal data, forecasting, and anomaly detection.",
        seed_texts=[
            "Deep learning for time series forecasting and prediction",
            "Transformer models for temporal sequence modeling",
            "Anomaly detection in time series data",
            "Foundation models for time series analysis",
        ],
        related_arxiv_categories=["cs.LG", "stat.ML", "cs.AI"],
    ),
    ConceptTag(
        key="causal_inference",
        label="Causal Inference",
        description="Causal discovery, treatment effect estimation, and causal representation learning.",
        seed_texts=[
            "Causal inference and treatment effect estimation from observational data",
            "Causal discovery algorithms and structure learning",
            "Causal representation learning and disentanglement",
            "Counterfactual reasoning and causal machine learning",
        ],
        related_arxiv_categories=["stat.ML", "cs.LG", "stat.ME", "cs.AI"],
    ),
    ConceptTag(
        key="speech_audio",
        label="Speech & Audio",
        description="Speech recognition, synthesis, music generation, and audio understanding.",
        seed_texts=[
            "End-to-end speech recognition and automatic speech understanding",
            "Text-to-speech synthesis and voice conversion",
            "Music generation and audio source separation",
            "Self-supervised learning for speech and audio representations",
        ],
        related_arxiv_categories=["cs.SD", "eess.AS", "cs.CL", "cs.LG"],
    ),
    ConceptTag(
        key="3d_vision",
        label="3D Vision & NeRF",
        description="Neural radiance fields, 3D reconstruction, and 3D generation.",
        seed_texts=[
            "Neural radiance fields for novel view synthesis",
            "3D Gaussian splatting and real-time neural rendering",
            "3D reconstruction from images and point clouds",
            "3D-aware image generation and shape modeling",
        ],
        related_arxiv_categories=["cs.CV", "cs.GR", "cs.LG"],
    ),
    ConceptTag(
        key="quantum_computing",
        label="Quantum Computing",
        description="Quantum algorithms, quantum machine learning, and quantum error correction.",
        seed_texts=[
            "Quantum machine learning and variational quantum algorithms",
            "Quantum error correction and fault-tolerant computation",
            "Quantum simulation of physical and chemical systems",
            "Hybrid classical-quantum computing and quantum advantage",
        ],
        related_arxiv_categories=["quant-ph", "cs.LG", "physics.comp-ph"],
    ),
    ConceptTag(
        key="financial_ml",
        label="Finance & Economics AI",
        description="ML for trading, risk, portfolio optimization, and economic modeling.",
        seed_texts=[
            "Machine learning for financial time series and stock prediction",
            "Deep learning for portfolio optimization and risk management",
            "Natural language processing for financial documents and sentiment",
            "Reinforcement learning for algorithmic trading strategies",
        ],
        related_arxiv_categories=["q-fin.CP", "q-fin.ST", "cs.LG", "stat.ML"],
    ),
    ConceptTag(
        key="ai_education",
        label="AI for Education",
        description="Intelligent tutoring, knowledge tracing, and educational data mining.",
        seed_texts=[
            "Intelligent tutoring systems and adaptive learning",
            "Knowledge tracing and student performance prediction",
            "Natural language processing for educational content and assessment",
            "Large language models for personalized education",
        ],
        related_arxiv_categories=["cs.CY", "cs.AI", "cs.CL", "cs.LG"],
    ),
]

CONCEPT_TAG_MAP: dict[str, ConceptTag] = {tag.key: tag for tag in CONCEPT_TAGS}
CONCEPT_EMBEDDINGS_FILE = "concept_embeddings.npy"
CONCEPT_EMBEDDINGS_META_FILE = "concept_embeddings_meta.json"

# Broad concepts that should contribute to threads but not easily create their own.
BROAD_CONCEPT_KEYS: set[str] = set()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a single vector."""
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of a matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def compute_concept_embeddings(model: object) -> dict[str, np.ndarray]:
    """Compute and store embeddings for all concept tags.

    Each tag's ``seed_texts`` are embedded as pseudo-papers (title-only,
    empty abstract), averaged, and L2-normalized.

    Args:
        model: An :class:`~pipeline.embed.EmbeddingModel` instance with an
            ``embed_papers`` method.

    Returns:
        Dict mapping concept key to unit-norm (768,) embedding.
    """
    all_papers: list[dict] = []
    tag_ranges: list[tuple[str, int, int]] = []

    start = 0
    for tag in CONCEPT_TAGS:
        papers = [{"title": text, "abstract": ""} for text in tag.seed_texts]
        all_papers.extend(papers)
        tag_ranges.append((tag.key, start, start + len(papers)))
        start += len(papers)

    all_embs = model.embed_papers(all_papers)  # (total_texts, 768)
    all_embs = _normalize_rows(all_embs)

    result: dict[str, np.ndarray] = {}
    for key, lo, hi in tag_ranges:
        avg = all_embs[lo:hi].mean(axis=0)
        emb = _normalize(avg).astype(np.float32)
        CONCEPT_TAG_MAP[key].embedding = emb
        result[key] = emb

    return result


def _ordered_concept_keys(embeddings: dict[str, np.ndarray]) -> list[str]:
    """Return concept keys in registry order, omitting missing embeddings."""
    return [tag.key for tag in CONCEPT_TAGS if tag.key in embeddings]


def save_concept_embedding_artifacts(
    embeddings: dict[str, np.ndarray],
    data_dir: str | Path = "data",
    metadata: dict | None = None,
) -> None:
    """Save concept embeddings as a compact matrix plus JSON metadata.

    Args:
        embeddings: Mapping from concept key to unit-norm embedding vector.
        data_dir: Directory where artifacts should be written.
        metadata: Optional extra metadata to merge into the JSON sidecar.

    Raises:
        ValueError: If embeddings are missing, malformed, non-finite, or not
            approximately unit-normalized.
    """
    keys = _ordered_concept_keys(embeddings)
    if not keys:
        raise ValueError("No concept embeddings were provided.")

    matrix = np.stack([np.asarray(embeddings[key], dtype=np.float32) for key in keys])
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Concept embeddings contain non-finite values.")

    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Concept embeddings must be unit-normalized.")

    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / CONCEPT_EMBEDDINGS_FILE, matrix.astype(np.float32))
    meta = {
        "keys": keys,
        "embedding_dim": int(matrix.shape[1]),
        "concept_count": int(matrix.shape[0]),
        "dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta.update(metadata)

    with open(output_dir / CONCEPT_EMBEDDINGS_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_concept_embedding_artifacts(
    data_dir: str | Path = "data",
) -> dict[str, np.ndarray]:
    """Load precomputed concept embeddings from disk.

    Returns:
        Dict mapping concept key to unit-norm float32 embedding.

    Raises:
        FileNotFoundError: If either artifact is missing.
        ValueError: If artifact contents are malformed.
    """
    data = Path(data_dir)
    matrix_path = data / CONCEPT_EMBEDDINGS_FILE
    meta_path = data / CONCEPT_EMBEDDINGS_META_FILE

    missing = [str(path) for path in (matrix_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing concept embedding artifact(s): " + ", ".join(missing)
        )

    matrix = np.load(matrix_path).astype(np.float32, copy=False)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    keys = meta.get("keys")
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise ValueError("Concept embedding metadata must contain a string 'keys' list.")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D concept embedding matrix, got {matrix.shape}.")
    if len(keys) != matrix.shape[0]:
        raise ValueError(
            f"Concept metadata key count {len(keys)} does not match matrix rows "
            f"{matrix.shape[0]}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Concept embeddings contain non-finite values.")

    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Concept embeddings must be unit-normalized.")

    result = {key: matrix[i].copy() for i, key in enumerate(keys)}
    for key, embedding in result.items():
        if key in CONCEPT_TAG_MAP:
            CONCEPT_TAG_MAP[key].embedding = embedding
    return result
