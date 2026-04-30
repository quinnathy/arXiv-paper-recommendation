"""Domain-specific joke boards for personalized daily feed flavor.

Each joke board represents a broad research domain with seed texts
(for embedding) and two one-liner jokes.  At runtime the board closest
to the user's centroids is selected, and one joke is chosen stably
per user per day.

Embeddings are precomputed offline via ``scripts/build_joke_embeddings.py``
and stored as ``data/joke_embeddings.npy`` + ``data/joke_embeddings_meta.json``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data model & registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JokeBoard:
    key: str
    label: str
    seed_texts: tuple[str, ...]
    jokes: tuple[str, str]


JOKE_BOARDS: tuple[JokeBoard, ...] = (
    JokeBoard(
        key="ml_dl",
        label="Machine Learning",
        seed_texts=(
            "Deep learning and neural network training optimization",
            "Supervised and unsupervised machine learning methods",
            "Gradient-based optimization for deep neural networks",
        ),
        jokes=(
            "\u201cIt\u2019s not overfitting; it\u2019s a long-term relationship with the training set.\u201d",
            "\u201cMy model generalized beautifully to the exact dataset I had in mind.\u201d",
        ),
    ),
    JokeBoard(
        key="llm_nlp",
        label="LLMs & NLP",
        seed_texts=(
            "Large language model pre-training and instruction tuning",
            "Natural language processing and text understanding",
            "Retrieval-augmented generation and language model reasoning",
        ),
        jokes=(
            "\u201cI asked for a citation and got a very fluent hallucination.\u201d",
            "\u201cNLP is easy, as long as nobody uses language creatively.\u201d",
        ),
    ),
    JokeBoard(
        key="cv_3d",
        label="Computer Vision",
        seed_texts=(
            "Image recognition and object detection with deep learning",
            "3D reconstruction and neural radiance fields",
            "Visual scene understanding and semantic segmentation",
        ),
        jokes=(
            "\u201cThe model understands the image perfectly, except for shadows, reflections, and reality.\u201d",
            "\u201cComputer vision works until the lighting has an opinion.\u201d",
        ),
    ),
    JokeBoard(
        key="recsys_search",
        label="Recommender Systems",
        seed_texts=(
            "Collaborative filtering and recommendation algorithms",
            "Information retrieval and search ranking systems",
            "Web-scale data mining and personalization",
        ),
        jokes=(
            "\u201cYou clicked one paper on graph neural networks, so we rebuilt your identity.\u201d",
            "\u201cSearch ranking is just relevance, popularity, and a small ritual sacrifice to latency.\u201d",
        ),
    ),
    JokeBoard(
        key="robotics",
        label="Robotics",
        seed_texts=(
            "Robot learning for manipulation and locomotion",
            "Embodied AI agents in simulated environments",
            "Sim-to-real transfer for robotic control",
        ),
        jokes=(
            "\u201cIt works in simulation, which is robotics for \u2018please don\u2019t ask about the demo.\u2019\u201d",
            "\u201cThe robot is autonomous until it meets a cable, a corner, or a chair.\u201d",
        ),
    ),
    JokeBoard(
        key="healthcare_ai",
        label="Healthcare AI",
        seed_texts=(
            "Machine learning for clinical decision support",
            "Deep learning for medical imaging and diagnosis",
            "Predictive modeling with electronic health records",
        ),
        jokes=(
            "\u201cThe AUC is excellent; now we just need the hospital Wi-Fi to cooperate.\u201d",
            "\u201cMedical AI: where the model is confident and the clinician still has to sign.\u201d",
        ),
    ),
    JokeBoard(
        key="comp_bio",
        label="Computational Biology",
        seed_texts=(
            "Computational biology and bioinformatics methods",
            "Single-cell genomics and gene expression analysis",
            "Protein structure prediction and biological sequence modeling",
        ),
        jokes=(
            "\u201cThe signal is strong, unless it\u2019s a batch effect wearing a lab coat.\u201d",
            "\u201cComputational biology: where every dataset comes with a hidden experimental design.\u201d",
        ),
    ),
    JokeBoard(
        key="drug_discovery",
        label="Drug Discovery",
        seed_texts=(
            "Machine learning for drug discovery and molecular generation",
            "Virtual screening and structure-based drug design",
            "Graph neural networks for molecular property prediction",
        ),
        jokes=(
            "\u201cThe molecule looks promising, which means biology has not rejected it yet.\u201d",
            "\u201cDrug discovery is just optimization with more ways to be humbled.\u201d",
        ),
    ),
    JokeBoard(
        key="comp_neuro",
        label="Computational Neuroscience",
        seed_texts=(
            "Computational models of neural coding and brain function",
            "Neural data analysis and brain-computer interfaces",
            "Spiking neural networks and biological learning rules",
        ),
        jokes=(
            "\u201cWe modeled the brain, assuming the brain agreed to be linear today.\u201d",
            "\u201cThe neuron fired, the model fit, and the interpretation remains pending.\u201d",
        ),
    ),
    JokeBoard(
        key="climate_weather",
        label="Climate & Weather",
        seed_texts=(
            "Machine learning for weather forecasting and climate modeling",
            "Earth system science and meteorological data analysis",
            "Neural operators for physical forecasting systems",
        ),
        jokes=(
            "\u201cThe forecast was right, just at the wrong scale, time, and grid cell.\u201d",
            "\u201cWeather modeling: where chaos is not a bug, it\u2019s the problem statement.\u201d",
        ),
    ),
    JokeBoard(
        key="sciml_fluids",
        label="Scientific ML",
        seed_texts=(
            "Physics-informed neural networks for partial differential equations",
            "Neural operators for computational fluid dynamics",
            "Surrogate modeling for scientific simulation",
        ),
        jokes=(
            "\u201cThe neural net solved the PDE, but the boundary conditions are filing a complaint.\u201d",
            "\u201cScientific ML: replacing equations with parameters, then rediscovering the equations.\u201d",
        ),
    ),
    JokeBoard(
        key="physics_astro",
        label="Physics & Astrophysics",
        seed_texts=(
            "Machine learning for particle physics and cosmology",
            "Astrophysical simulation and gravitational wave analysis",
            "Data-driven methods for high-energy physics experiments",
        ),
        jokes=(
            "\u201cThe universe is expanding, and so is the error bar.\u201d",
            "\u201cAstrophysics: the lab is inconveniently far away.\u201d",
        ),
    ),
    JokeBoard(
        key="quantum",
        label="Quantum Computing",
        seed_texts=(
            "Quantum machine learning and variational quantum algorithms",
            "Quantum error correction and fault-tolerant computation",
            "Hybrid classical-quantum computing and quantum simulation",
        ),
        jokes=(
            "\u201cThe qubit was stable until someone wanted to use it.\u201d",
            "\u201cQuantum computing: great in theory, noisy in practice, magical in slides.\u201d",
        ),
    ),
    JokeBoard(
        key="math",
        label="Mathematics",
        seed_texts=(
            "Algebraic geometry and abstract algebra",
            "Number theory and combinatorial mathematics",
            "Proof assistants and formalized mathematics",
        ),
        jokes=(
            "\u201cThe proof is obvious after you already understand it.\u201d",
            "\u201cMathematicians call it trivial when they don\u2019t want to write three pages.\u201d",
        ),
    ),
    JokeBoard(
        key="statistics",
        label="Statistics",
        seed_texts=(
            "Bayesian inference and probabilistic modeling",
            "Statistical hypothesis testing and experimental design",
            "High-dimensional data analysis and nonparametric methods",
        ),
        jokes=(
            "\u201cStatistics means never having to say you\u2019re certain.\u201d",
            "\u201cThe p-value was significant, but my faith in the design was not.\u201d",
        ),
    ),
    JokeBoard(
        key="optimization",
        label="Optimization",
        seed_texts=(
            "Convex optimization and gradient descent methods",
            "Numerical linear algebra and iterative solvers",
            "Stochastic optimization and convergence analysis",
        ),
        jokes=(
            "\u201cGradient descent is just optimism with a step size.\u201d",
            "\u201cIt converged, but nobody asked where.\u201d",
        ),
    ),
    JokeBoard(
        key="algorithms",
        label="Algorithms & Theory",
        seed_texts=(
            "Algorithm design and computational complexity theory",
            "Approximation algorithms and combinatorial optimization",
            "Randomized algorithms and data structures",
        ),
        jokes=(
            "\u201cIt\u2019s polynomial time, but the polynomial has tenure.\u201d",
            "\u201cThe asymptotics are beautiful; the constants are seeking revenge.\u201d",
        ),
    ),
    JokeBoard(
        key="systems",
        label="Systems & Distributed Computing",
        seed_texts=(
            "Distributed systems and consensus protocols",
            "Operating system design and scheduling algorithms",
            "Cloud computing and large-scale systems infrastructure",
        ),
        jokes=(
            "\u201cDistributed systems: one computer fails, and somehow five computers are guilty.\u201d",
            "\u201cIt worked locally, which is the first stage of grief.\u201d",
        ),
    ),
    JokeBoard(
        key="databases",
        label="Databases & Data Mining",
        seed_texts=(
            "Database query optimization and indexing structures",
            "Data mining and knowledge discovery in large datasets",
            "Transaction processing and data warehouse systems",
        ),
        jokes=(
            "\u201cThe query is optimized, but the schema remembers every bad decision.\u201d",
            "\u201cData mining: finding patterns, then finding out they were preprocessing artifacts.\u201d",
        ),
    ),
    JokeBoard(
        key="networks",
        label="Computer Networks",
        seed_texts=(
            "Network protocol design and internet architecture",
            "Software-defined networking and traffic engineering",
            "Wireless communication and mobile network optimization",
        ),
        jokes=(
            "\u201cI\u2019d tell you a UDP joke, but you might not get it.\u201d",
            "\u201cNetworking is easy until the packet takes a scenic route.\u201d",
        ),
    ),
    JokeBoard(
        key="security",
        label="Security & Cryptography",
        seed_texts=(
            "Cryptographic protocols and secure computation",
            "Privacy-preserving machine learning and differential privacy",
            "Network security and adversarial attack detection",
        ),
        jokes=(
            "\u201cThe protocol is secure, assuming users behave like the proof.\u201d",
            "\u201cPrivacy-preserving ML: hiding the data while trying not to hide the signal.\u201d",
        ),
    ),
    JokeBoard(
        key="hci_education",
        label="HCI & AI for Education",
        seed_texts=(
            "Human-computer interaction and user interface design",
            "Intelligent tutoring systems and adaptive learning",
            "AI for education and personalized learning platforms",
        ),
        jokes=(
            "\u201cThe interface is intuitive to everyone who attended the design meeting.\u201d",
            "\u201cAI for education: personalized learning, standardized testing, and one confused dashboard.\u201d",
        ),
    ),
    JokeBoard(
        key="responsible_ai",
        label="Responsible AI",
        seed_texts=(
            "Algorithmic fairness and bias mitigation in machine learning",
            "AI safety and value alignment for autonomous systems",
            "Interpretability and explainability of neural network decisions",
        ),
        jokes=(
            "\u201cThe benchmark says it\u2019s fair; society has follow-up questions.\u201d",
            "\u201cAI safety: discovering edge cases before the edge cases discover us.\u201d",
        ),
    ),
    JokeBoard(
        key="econ_finance",
        label="Economics & Finance",
        seed_texts=(
            "Machine learning for financial time series and trading",
            "Game theory and mechanism design in economics",
            "Algorithmic trading and portfolio optimization",
        ),
        jokes=(
            "\u201cThe market is efficient, except immediately after I make a prediction.\u201d",
            "\u201cGame theory assumes everyone is rational, which is a bold onboarding choice.\u201d",
        ),
    ),
    JokeBoard(
        key="audio_speech",
        label="Audio & Speech",
        seed_texts=(
            "Automatic speech recognition and language understanding",
            "Text-to-speech synthesis and voice conversion",
            "Audio signal processing and music generation",
        ),
        jokes=(
            "\u201cThe signal is clean, except for the part people care about.\u201d",
            "\u201cSpeech recognition works great until someone speaks naturally.\u201d",
        ),
    ),
    JokeBoard(
        key="pl_formal",
        label="Programming Languages",
        seed_texts=(
            "Programming language theory and type systems",
            "Formal verification and program correctness proofs",
            "Static analysis and compiler optimization",
        ),
        jokes=(
            "\u201cThe type checker rejected my program but improved my character.\u201d",
            "\u201cFormal methods: because \u2018it seems to work\u2019 was not a proof.\u201d",
        ),
    ),
    JokeBoard(
        key="se_ai4code",
        label="Software Engineering",
        seed_texts=(
            "Automated software engineering and code generation",
            "Software testing and continuous integration",
            "AI-assisted code review and bug detection",
        ),
        jokes=(
            "\u201cThe code is self-documenting, but unfortunately very private.\u201d",
            "\u201cAI for code: now bugs can be generated at scale.\u201d",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Artifact file names
# ---------------------------------------------------------------------------

JOKE_EMBEDDINGS_FILE = "joke_embeddings.npy"
JOKE_EMBEDDINGS_META_FILE = "joke_embeddings_meta.json"


# ---------------------------------------------------------------------------
# Offline: compute & save
# ---------------------------------------------------------------------------

def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def compute_joke_embeddings(model: object) -> np.ndarray:
    """Embed all joke-board seed texts and return (N_boards, 768) matrix.

    Args:
        model: An :class:`~pipeline.embed.EmbeddingModel` instance.

    Returns:
        Unit-norm float32 matrix of shape ``(len(JOKE_BOARDS), dim)``.
    """
    all_papers: list[dict] = []
    ranges: list[tuple[int, int]] = []
    start = 0
    for board in JOKE_BOARDS:
        papers = [{"title": t, "abstract": ""} for t in board.seed_texts]
        all_papers.extend(papers)
        ranges.append((start, start + len(papers)))
        start += len(papers)

    raw = model.embed_papers(all_papers)
    raw = _normalize_rows(raw)

    board_embs: list[np.ndarray] = []
    for lo, hi in ranges:
        avg = raw[lo:hi].mean(axis=0)
        norm = np.linalg.norm(avg)
        board_embs.append(avg / max(norm, 1e-12))

    return np.asarray(board_embs, dtype=np.float32)


def save_joke_embedding_artifacts(
    matrix: np.ndarray,
    data_dir: str | Path = "data",
    metadata: dict | None = None,
) -> None:
    """Write joke embeddings to ``data_dir`` as .npy + .json sidecar."""
    keys = [b.key for b in JOKE_BOARDS]
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / JOKE_EMBEDDINGS_FILE, matrix.astype(np.float32))

    meta: dict = {
        "keys": keys,
        "embedding_dim": int(matrix.shape[1]),
        "board_count": int(matrix.shape[0]),
        "dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta.update(metadata)

    with open(out / JOKE_EMBEDDINGS_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Online: load from disk
# ---------------------------------------------------------------------------

def load_joke_embedding_artifacts(
    data_dir: str | Path = "data",
) -> np.ndarray | None:
    """Load precomputed joke embeddings.  Returns ``None`` if missing."""
    npy = Path(data_dir) / JOKE_EMBEDDINGS_FILE
    if not npy.exists():
        return None
    try:
        matrix = np.load(npy).astype(np.float32, copy=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(JOKE_BOARDS):
            return None
        return matrix
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _pick_board_and_joke(
    user_centroids: np.ndarray,
    joke_embeddings: np.ndarray,
    user_id: str,
    today: date,
) -> dict[str, str]:
    """Pure selection logic (no Streamlit dependency, easy to unit-test).

    Returns ``{"label": ..., "joke": ...}``.
    """
    # (k_u, 768) @ (768, N_boards) → (k_u, N_boards)
    sims = np.asarray(user_centroids, dtype=np.float32) @ joke_embeddings.T
    best_idx = int(sims.max(axis=0).argmax())
    board = JOKE_BOARDS[best_idx]

    # Stable daily pick: SHA-256(user_id:date) → index into the two jokes
    h = hashlib.sha256(f"{user_id}:{today.isoformat()}".encode()).hexdigest()
    joke_idx = int(h, 16) % 2

    return {"label": board.label, "joke": board.jokes[joke_idx]}


def select_domain_joke(
    user_centroids: np.ndarray,
    user_id: str,
    today: date | None = None,
) -> dict[str, str] | None:
    """Select a domain-specific joke for the daily feed.

    Returns ``{"label": ..., "joke": ...}`` or *None* if anything fails.
    Loads precomputed embeddings from ``data/``; returns *None* if the
    artifact has not been built yet.
    """
    if user_centroids is None or len(user_centroids) == 0:
        return None

    joke_embs = load_joke_embedding_artifacts()
    if joke_embs is None:
        return None

    try:
        return _pick_board_and_joke(
            user_centroids, joke_embs, user_id, today or date.today()
        )
    except Exception:
        return None


def _available_loading_boards(data_dir: str | Path = "data") -> list[JokeBoard]:
    meta_path = Path(data_dir) / JOKE_EMBEDDINGS_META_FILE
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            keys = set(json.load(fh).get("keys", []))
        saved_boards = [board for board in JOKE_BOARDS if board.key in keys]
        return saved_boards
    except Exception:
        return []


def random_loading_joke(data_dir: str | Path = "data") -> str:
    """Return a random joke from locally available joke metadata."""
    boards = _available_loading_boards(data_dir)
    if not boards:
        return ""

    board = random.choice(boards)
    return random.choice(board.jokes)


def random_loading_jokes(
    count: int = 5,
    data_dir: str | Path = "data",
) -> list[str]:
    """Return a small rotating set of local jokes for loading states."""
    boards = _available_loading_boards(data_dir)
    if not boards:
        return []

    jokes = [joke for board in boards for joke in board.jokes]
    if not jokes:
        return []

    count = max(1, count)
    return [random.choice(jokes) for _ in range(count)]
