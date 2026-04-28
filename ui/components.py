"""Reusable Streamlit UI widgets for the ArXiv Daily app.

Provides self-contained components that can be composed into pages:
- paper_card: renders a single paper with action buttons.
- unified_tag_selector: single multiselect for arXiv categories + concept tags.
- free_text_input: text area for free-form research interests.
- loading_spinner_with_message: context manager for custom spinner messages.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import streamlit as st

from pipeline.concept_tags import CONCEPT_TAG_MAP


# Human-readable onboarding label -> one or more arXiv category codes.
# Only categories present in category_centroids are used at runtime.
TOPIC_LABELS: dict[str, list[str]] = {
    # AI / ML core
    "Artificial Intelligence": ["cs.AI"],
    "Machine Learning": ["cs.LG", "stat.ML"],
    "Deep Learning & Neural Networks": ["cs.LG", "cs.NE", "stat.ML"],
    "Reinforcement Learning & Decision Making": [
        "cs.LG",
        "cs.AI",
        "cs.MA",
        "eess.SY",
    ],
    "Natural Language Processing": ["cs.CL", "cs.LG", "cs.IR"],
    "Large Language Models": ["cs.CL", "cs.LG", "cs.AI", "cs.IR"],
    "Computer Vision": ["cs.CV", "cs.LG", "eess.IV"],
    "Generative Models": ["cs.LG", "cs.CV", "cs.CL", "stat.ML"],

    # AI applications / interdisciplinary ML
    "Healthcare AI": [
        "cs.LG",
        "cs.CV",
        "cs.CL",
        "cs.AI",
        "stat.ML",
        "q-bio.QM",
        "eess.IV",
    ],
    "Medical Imaging": ["cs.CV", "eess.IV", "cs.LG", "stat.ML", "q-bio.QM"],
    "Computational Biology & Bioinformatics": [
        "q-bio.QM",
        "q-bio.GN",
        "q-bio.MN",
        "q-bio.BM",
        "cs.LG",
        "stat.ML",
    ],
    "Computational Neuroscience": ["q-bio.NC", "cs.NE", "cs.LG", "stat.ML"],
    "Climate, Weather & Earth Systems": [
        "physics.ao-ph",
        "physics.geo-ph",
        "cs.LG",
        "stat.ML",
        "math.NA",
    ],
    "Scientific Machine Learning": [
        "cs.LG",
        "stat.ML",
        "math.NA",
        "math.OC",
        "physics.comp-ph",
        "cs.CE",
    ],
    "AI for Education": ["cs.CY", "cs.HC", "cs.AI", "cs.LG", "cs.CL"],
    "Finance & Economics AI": [
        "q-fin.CP",
        "q-fin.ST",
        "q-fin.PM",
        "q-fin.RM",
        "econ.EM",
        "cs.LG",
        "stat.ML",
    ],

    # Data / information / retrieval
    "Information Retrieval & Search": ["cs.IR", "cs.CL", "cs.DL", "cs.DB"],
    "Databases & Data Mining": ["cs.DB", "cs.LG", "stat.ML"],
    "Recommender Systems & Web Data": ["cs.IR", "cs.SI", "cs.LG", "cs.DB"],
    "Social & Information Networks": ["cs.SI", "cs.CY", "cs.LG", "stat.ML"],

    # Human-centered computing
    "Human-Computer Interaction": ["cs.HC", "cs.CY"],
    "Computers and Society": ["cs.CY", "cs.HC", "cs.SI"],
    "Responsible AI, Fairness & Society": ["cs.CY", "cs.AI", "cs.LG", "stat.ML"],

    # Robotics / control / embodied systems
    "Robotics": ["cs.RO", "cs.AI", "cs.LG", "eess.SY"],
    "Control Systems": ["eess.SY", "math.OC", "cs.SY"],
    "Multiagent Systems": ["cs.MA", "cs.AI", "cs.GT", "cs.LG"],

    # Systems / software / infrastructure
    "Distributed & Parallel Computing": ["cs.DC", "cs.PF", "cs.OS"],
    "Computer Networks": ["cs.NI", "cs.DC"],
    "Operating Systems": ["cs.OS", "cs.DC", "cs.PF"],
    "Software Engineering": ["cs.SE", "cs.PL"],
    "Programming Languages": ["cs.PL", "cs.LO"],
    "Computer Architecture": ["cs.AR", "cs.ET"],
    "Security & Privacy": ["cs.CR", "cs.CY"],

    # Theory / algorithms / formal methods
    "Algorithms & Data Structures": ["cs.DS", "cs.CC"],
    "Computational Complexity": ["cs.CC", "cs.DS"],
    "Logic & Formal Methods": ["cs.LO", "math.LO", "cs.FL"],
    "Cryptography Theory": ["cs.CR", "cs.IT", "math.NT"],
    "Game Theory & Mechanism Design": ["cs.GT", "econ.TH", "math.OC"],

    # Statistics / probability / optimization
    "Statistics & Data Analysis": ["stat.AP", "stat.ME", "stat.CO", "math.ST"],
    "Statistical Machine Learning": ["stat.ML", "cs.LG", "stat.ME"],
    "Probability & Stochastic Processes": ["math.PR", "stat.TH"],
    "Optimization": ["math.OC", "cs.LG", "stat.ML"],

    # Mathematics
    "Pure Mathematics": [
        "math.AG",
        "math.AT",
        "math.CO",
        "math.DG",
        "math.FA",
        "math.GT",
        "math.NT",
        "math.RT",
    ],
    "Applied Mathematics": ["math.NA", "math.OC", "math.AP", "math.DS", "math.PR"],
    "Numerical Analysis & Scientific Computing": [
        "math.NA",
        "cs.NA",
        "cs.CE",
        "physics.comp-ph",
    ],
    "Dynamical Systems": ["math.DS", "math.OC", "physics.class-ph"],

    # Physics / astronomy / quantum
    "Computational Physics": ["physics.comp-ph", "cs.CE", "math.NA"],
    "Quantum Information & Quantum Computing": ["quant-ph", "cs.ET", "cs.IT"],
    "Astrophysics & Cosmology": [
        "astro-ph.CO",
        "astro-ph.GA",
        "astro-ph.HE",
        "astro-ph.IM",
        "astro-ph.SR",
        "astro-ph.EP",
    ],
    "Condensed Matter Physics": [
        "cond-mat.mtrl-sci",
        "cond-mat.stat-mech",
        "cond-mat.mes-hall",
        "cond-mat.soft",
        "cond-mat.str-el",
        "cond-mat.supr-con",
    ],
    "High Energy Physics": ["hep-th", "hep-ph", "hep-ex", "hep-lat"],
    "Nuclear Physics": ["nucl-th", "nucl-ex"],
    "Fluid Dynamics": ["physics.flu-dyn", "nlin.CD", "math.AP"],
    "Optics & Photonics": ["physics.optics", "eess.SP"],

    # Signal / image / audio
    "Signal Processing": ["eess.SP", "cs.IT", "stat.ML"],
    "Image & Video Processing": ["eess.IV", "cs.CV", "cs.MM"],
    "Audio & Speech Processing": ["eess.AS", "cs.SD", "cs.CL"],

    # Quantitative finance / economics
    "Quantitative Finance": [
        "q-fin.CP",
        "q-fin.MF",
        "q-fin.PM",
        "q-fin.PR",
        "q-fin.RM",
        "q-fin.ST",
        "q-fin.TR",
    ],
    "Economics": ["econ.EM", "econ.GN", "econ.TH"],
}


def available_topic_labels(
    topic_labels: dict[str, list[str]],
    category_centroids: dict[str, np.ndarray],
) -> list[str]:
    """Return onboarding labels with at least one available arXiv category."""
    return [
        label
        for label, cats in topic_labels.items()
        if any(cat in category_centroids for cat in cats)
    ]


def expand_topic_labels(
    selected_labels: list[str],
    topic_labels: dict[str, list[str]],
    category_centroids: dict[str, np.ndarray],
) -> list[str]:
    """Expand selected onboarding labels into available arXiv category codes."""
    expanded: list[str] = []
    seen: set[str] = set()

    for label in selected_labels:
        for cat in topic_labels.get(label, []):
            if cat in category_centroids and cat not in seen:
                expanded.append(cat)
                seen.add(cat)

    return expanded


def paper_card(
    meta: dict,
    on_like: Callable[[str], None],
    on_save: Callable[[str], None],
    on_skip: Callable[[str], None],
) -> None:
    """Render a single paper as a Streamlit card with action buttons.

    Args:
        meta: Paper metadata dict with keys: id, title, abstract, categories,
            update_date, cluster_id, rec_score.
        on_like: Callback invoked with arxiv_id when Like is clicked.
        on_save: Callback invoked with arxiv_id when Save is clicked.
        on_skip: Callback invoked with arxiv_id when Skip is clicked.
    """
    arxiv_id = meta["id"]
    responded = st.session_state.get("responded", set())
    is_responded = arxiv_id in responded

    with st.container(border=True):
        title_text = " ".join(meta["title"].split())
        st.markdown(
            f'<p style="font-size:3.0rem; font-weight:650; '
            f'line-height:1.5; margin-bottom:0.30rem;">{title_text}</p>',
            unsafe_allow_html=True,
        )

        # Category badges
        categories = meta.get("categories", [])
        if categories:
            st.caption(" ".join(f"`{cat}`" for cat in categories))

        # Abstract snippet
        abstract = meta.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        st.write(abstract)

        # Links
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        st.markdown(f"[ArXiv page]({abs_url}) | [PDF]({pdf_url})")

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "Like" if not is_responded else "Liked",
                key=f"like_{arxiv_id}",
                disabled=is_responded,
            ):
                on_like(arxiv_id)
        with col2:
            if st.button(
                "Save" if not is_responded else "Saved",
                key=f"save_{arxiv_id}",
                disabled=is_responded,
            ):
                on_save(arxiv_id)
        with col3:
            if st.button(
                "Skip" if not is_responded else "Skipped",
                key=f"skip_{arxiv_id}",
                disabled=is_responded,
            ):
                on_skip(arxiv_id)


def unified_tag_selector(
    category_centroids: dict[str, np.ndarray],
    concept_embeddings: dict[str, np.ndarray],
) -> tuple[list[str], list[str]]:
    """Single multiselect combining arXiv categories and concept tags.

    Concept tags take priority when a label collision occurs (e.g.
    "Computational Biology" exists in both pools).

    Args:
        category_centroids: Dict mapping arXiv category code to centroid vector.
        concept_embeddings: Dict mapping concept key to unit-norm embedding.

    Returns:
        ``(selected_topic_labels, selected_concept_keys)`` — two lists
        partitioned by source so the caller can expand labels and build the
        right seed type.
    """
    # label → ("concept", concept_key) or ("category", onboarding_label)
    label_map: dict[str, tuple[str, str]] = {}

    # Concept tags first — they win on label collisions.
    for key, tag in CONCEPT_TAG_MAP.items():
        if key in concept_embeddings:
            label_map[tag.label] = ("concept", key)

    # arXiv-backed onboarding topics — skip labels already claimed by a concept tag.
    for label in available_topic_labels(TOPIC_LABELS, category_centroids):
        if label not in label_map:
            label_map[label] = ("category", label)

    options = list(label_map.keys())

    # Stable random order within the session
    if "tag_order_seed" not in st.session_state:
        st.session_state["tag_order_seed"] = random.randint(0, 2**31)
    rng = random.Random(st.session_state["tag_order_seed"])
    rng.shuffle(options)

    # Assign a stable color to each option (seeded by tag_order_seed)
    _PILL_PALETTE = [
        "#D69399", "#648C64", "#889EBA", "#C1AB7C",
        "#9C85A7", "#9EECD4", "#D28C5E", "#89A8B6",
        "#BE88A4", "#7AB196", "#7F609B", "#FCE88E",
    ]
    color_rng = random.Random(st.session_state["tag_order_seed"] + 1)
    option_colors = [color_rng.choice(_PILL_PALETTE) for _ in options]

    # Build per-pill CSS rules — use descendant selectors (Streamlit wraps
    # the tablist in an extra <div> so direct-child `>` doesn't reach it).
    pill_rules = []
    for i, color in enumerate(option_colors):
        pill_rules.append(
            f"div[data-testid='stPills'] div[role='tablist'] "
            f"button:nth-child({i + 1}):not([aria-checked='true']) "
            f"{{ background-color: {color} !important; "
            f"border-color: {color} !important; }}"
        )

    # Slow horizontal scroll animation for the pill container
    scroll_css = """
    @keyframes pill-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    div[data-testid='stPills'] div[role='tablist'] {
        animation: pill-scroll 60s linear infinite;
        width: max-content !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid='stPills'] div[role='tablist']:hover {
        animation-play-state: paused;
    }
    div[data-testid='stPills'] {
        overflow: hidden !important;
    }
    """

    st.markdown(
        f"<style>{scroll_css}\n{''.join(pill_rules)}</style>",
        unsafe_allow_html=True,
    )

    selected_labels = st.pills(
        "Select your research interests",
        options=options,
        selection_mode="multi",
        default=None,
        label_visibility="collapsed",
    )

    if not selected_labels:
        selected_labels = []

    # Partition selections back into topic labels and concepts.
    topic_labels: list[str] = []
    concept_keys: list[str] = []
    for label in selected_labels:
        kind, key = label_map[label]
        if kind == "concept":
            concept_keys.append(key)
        else:
            topic_labels.append(key)

    return topic_labels, concept_keys


def free_text_input() -> list[str]:
    """Render a text area for free-form research interest descriptions.

    Returns:
        List of non-empty phrase strings (one per line).
    """
    st.write("**Describe your interests in your own words** (optional)")
    raw = st.text_area(
        "Free-text interests",
        placeholder=(
            "e.g., diffusion models for medical imaging\n"
            "single-cell perturbation modeling\n"
            "LLMs for clinical decision support"
        ),
        label_visibility="collapsed",
    )

    if not raw or not raw.strip():
        return []

    phrases = [p.strip() for p in raw.splitlines()]
    return [p for p in phrases if p]


def loading_spinner_with_message(message: str):
    """Context manager wrapping st.spinner with a custom message.

    Args:
        message: The message to display inside the spinner.

    Returns:
        A context manager (st.spinner instance).
    """
    return st.spinner(message)
