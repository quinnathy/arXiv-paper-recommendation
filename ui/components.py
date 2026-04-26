"""Reusable Streamlit UI widgets for the ArXiv Daily app.

Provides self-contained components that can be composed into pages:
- paper_card: renders a single paper with action buttons.
- unified_tag_selector: single multiselect for arXiv categories + concept tags.
- free_text_input: text area for free-form research interests.
- loading_spinner_with_message: context manager for custom spinner messages.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import streamlit as st

from pipeline.concept_tags import CONCEPT_TAG_MAP


# Human-readable label -> arXiv category code mapping.
# Only topics that exist in the corpus (category_centroids) will be shown.
TOPIC_LABELS: dict[str, str] = {
    "Machine Learning": "cs.LG",
    "Computer Vision": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Robotics": "cs.RO",
    "Statistics / ML Theory": "stat.ML",
    "Artificial Intelligence": "cs.AI",
    "Computation & Language": "cs.CL",
    "Neural Networks": "cs.NE",
    "Information Retrieval": "cs.IR",
    "Human-Computer Interaction": "cs.HC",
    "Cryptography & Security": "cs.CR",
    "Distributed Computing": "cs.DC",
    "Computational Biology": "q-bio.QM",
    "Physics & ML": "physics.comp-ph",
    "Quantitative Finance": "q-fin.CP",
}


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
        st.subheader(meta["title"])

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
        ``(selected_category_codes, selected_concept_keys)`` — two lists
        partitioned by source so the caller can build the right seed type.
    """
    # label → ("concept", concept_key) or ("category", arxiv_code)
    label_map: dict[str, tuple[str, str]] = {}

    # Concept tags first — they win on label collisions.
    for key, tag in CONCEPT_TAG_MAP.items():
        if key in concept_embeddings:
            label_map[tag.label] = ("concept", key)

    # arXiv categories — skip labels already claimed by a concept tag.
    for label, code in TOPIC_LABELS.items():
        if code in category_centroids and label not in label_map:
            label_map[label] = ("category", code)

    options = sorted(label_map.keys())

    selected_labels = st.multiselect(
        "Select your research interests",
        options=options,
        default=None,
    )

    # Partition selections back into categories and concepts.
    category_codes: set[str] = set()
    concept_keys: list[str] = []
    for label in selected_labels:
        kind, key = label_map[label]
        if kind == "concept":
            concept_keys.append(key)
        else:
            category_codes.add(key)  # dedup (e.g. cs.CL appears twice)

    return list(category_codes), concept_keys


def free_text_input() -> list[str]:
    """Render a text area for free-form research interest descriptions.

    Returns:
        List of non-empty phrase strings (one per line).
    """
    with st.expander("Describe your interests in your own words (optional)"):
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
