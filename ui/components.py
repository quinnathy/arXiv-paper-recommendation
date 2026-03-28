"""Reusable Streamlit UI widgets for the ArXiv Daily app.

Provides self-contained components that can be composed into pages:
- paper_card: renders a single paper with action buttons.
- topic_selector: multiselect for arXiv categories with human-readable labels.
- loading_spinner_with_message: context manager for custom spinner messages.
"""

from __future__ import annotations

from typing import Callable


# Human-readable label → arXiv category code mapping.
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

    Layout:
        - Title (st.subheader)
        - Category badges (st.caption with colored tags)
        - Abstract snippet: first 300 characters + "..."
        - Links: [ArXiv page] [PDF]
        - Three action buttons in columns: Like | Save | Skip

    Args:
        meta: Paper metadata dict with keys: id, title, abstract, categories,
            update_date, cluster_id, rec_score.
        on_like: Callback invoked with arxiv_id when Like is clicked.
        on_save: Callback invoked with arxiv_id when Save is clicked.
        on_skip: Callback invoked with arxiv_id when Skip is clicked.

    Implementation:
        - Use st.container() for card grouping.
        - Display title with st.subheader.
        - Show categories as a comma-separated st.caption.
        - Truncate abstract to 300 chars + "..." for the snippet.
        - Build ArXiv URL: https://arxiv.org/abs/{id}
        - Build PDF URL: https://arxiv.org/pdf/{id}
        - Render Like/Save/Skip as st.button in 3 columns.
        - Each button calls its respective callback with meta["id"].
    """
    raise NotImplementedError


def topic_selector(category_centroids: dict) -> list[str]:
    """Render a multiselect widget for arXiv topic categories.

    Only shows topics that exist in the loaded corpus (i.e., topics whose
    arXiv category code is a key in category_centroids).

    Args:
        category_centroids: Dict mapping arXiv category string to centroid
            vector. Used to filter TOPIC_LABELS to only available topics.

    Returns:
        List of selected arXiv category code strings, e.g. ["cs.LG", "cs.CV"].

    Implementation:
        - Filter TOPIC_LABELS to only include labels whose category code
          exists as a key in category_centroids.
        - Display st.multiselect with the filtered human-readable labels.
        - Map selected labels back to arXiv category codes.
        - Return the list of category codes.
    """
    raise NotImplementedError


def loading_spinner_with_message(message: str):
    """Context manager wrapping st.spinner with a custom message.

    Args:
        message: The message to display inside the spinner.

    Returns:
        A context manager (st.spinner instance).

    Usage:
        with loading_spinner_with_message("Loading papers..."):
            do_something()
    """
    raise NotImplementedError
