"""Reusable Streamlit UI widgets for the ArXiv Daily app.

Provides self-contained components that can be composed into pages:
- paper_card: renders a single paper with action buttons.
- topic_selector: multiselect for arXiv categories with human-readable labels.
- loading_spinner_with_message: context manager for custom spinner messages.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st


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
    arxiv_id = meta["id"]
    responded = st.session_state.get("responded", set())
    is_responded = arxiv_id in responded

    # Use a custom div for the whole card
    st.markdown(f'<div class="arxiv-card">', unsafe_allow_html=True)
    
    with st.container(border=True):
        # We can still use native Streamlit components inside our styled container
        st.markdown(f'<h3 class="paper-title">{meta["title"]}</h3>', unsafe_allow_html=True)

        categories = meta.get("categories", [])
        if categories:
            st.caption(" ".join(f"`{cat}`" for cat in categories))

        abstract = meta.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        st.write(abstract)

        # Action row styling
        col1, col2, col3 = st.columns(3)
    
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
        
    st.markdown('</div>', unsafe_allow_html=True)

TOPIC_COLORS: dict[str, str] = {
    # BLUES (Primary Research & ML)
    "cs.LG": "#3498db",           # Bright Blue (Machine Learning)
    "cs.CL": "#2980b9",           # Medium Blue (NLP)
    "cs.AI": "#5dade2",           # Sky Blue (Artificial Intelligence)
    "cs.IR": "#21618c",           # Deep Navy (Information Retrieval)
    "physics.comp-ph": "#1f618d",  # Steel Blue (Physics & ML)
    
    # REDS/ORANGES (Vision & Robotics)
    "cs.CV": "#e74c3c",           # Bright Red (Computer Vision)
    "cs.RO": "#e67e22",           # Carrot Orange (Robotics)
    "cs.DC": "#d35400",           # Burnt Orange (Distributed Computing)
    
    # GREENS (Theory, Bio, and Language)
    "stat.ML": "#27ae60",         # Emerald Green (Statistics / ML Theory)
    "q-bio.QM": "#16a085",        # Sea Green (Computational Biology)
    # Note: Using a lime-tinted green here to distinguish from Stats

    # YELLOWS (Neural Nets)
    "cs.NE": "#f1c40f",           # Vivid Yellow (Neural Networks)
    
    # PURPLES (HCI & Finance)
    "cs.HC": "#8e44ad",           # Amethyst Purple (HCI)
    "q-fin.CP": "#7d3c98",        # Royal Purple (Quantitative Finance)
    
    # TEALS (Security)
    "cs.CR": "#138d75",           # Dark Teal (Cryptography & Security)
    
    "default": "#7f8c8d"          # Concrete Gray
}

def topic_selector(category_centroids: dict) -> list[str]:
    available = {label: code for label, code in TOPIC_LABELS.items() 
                 if code in category_centroids}
    
    if "selected_tags" not in st.session_state:
        st.session_state.selected_tags = set()

    # 1. Build the CSS and the Container
    html_lines = [
        "<style>",
        "  .tag-cloud { display: flex; justify-content: center;flex-wrap: wrap; gap: 12px; padding: 12px 0; }",
        "  .custom-tag {",
        "    border-radius: 20px; padding: 4px 12px; font-family: 'Arial', sans-serif;",
        "    font-weight: 900; font-size: 0.8rem; cursor: pointer;",
        "    border: 1.5px solid var(--c); background: transparent; color: var(--c) !important;",
        "    white-space: nowrap; transition: 0.2s;text-decoration: none !important; display: inline-block;",
        "    line-height: 1.2;",
        "  }",
        "  .custom-tag:hover { background: var(--c); color: white !important; }",
        "  .custom-tag.active { background: var(--c); color: white !important;; }",
        "</style>",
        '<div class="tag-cloud">'
    ]
    
    # 2. Generate Tag Links
    for label, code in available.items():
        is_selected = code in st.session_state.selected_tags
        color = TOPIC_COLORS.get(code, TOPIC_COLORS["default"])
        active_class = "active" if is_selected else ""
        
        # We use st.query_params logic: clicking the link reloads with ?topic=code
        tag_html = f'<a href="?topic={code}" target="_self" class="custom-tag {active_class}" style="--c: {color};">{label}</a>'
        html_lines.append(tag_html)
    
    html_lines.append("</div>")
    
    # 3. Render once as a single block
    st.markdown("\n".join(html_lines), unsafe_allow_html=True)

    # 4. Logic to catch the click
    if "topic" in st.query_params:
        clicked_code = st.query_params["topic"]
        if clicked_code in st.session_state.selected_tags:
            st.session_state.selected_tags.remove(clicked_code)
        else:
            st.session_state.selected_tags.add(clicked_code)
        
        # Clear params and rerun to update the UI
        st.query_params.clear()
        st.rerun()

    return list(st.session_state.selected_tags)

def loading_spinner_with_message(message: str):
    """Context manager wrapping st.spinner with a custom message.

    Args:
        message: The message to display inside the spinner.

    Returns:
        A context manager (st.spinner instance).
    """
    return st.spinner(message)

def inject_design():
    """Reads the local CSS file and injects it into the Streamlit app."""
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback if the file isn't found during dev
        pass