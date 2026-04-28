"""Archive page showing papers the user has saved."""

from __future__ import annotations

import streamlit as st

from user.db import get_saved_papers


def render_archive_page() -> None:
    """Render the saved-papers archive."""
    user_id = st.session_state["user_id"]
    saved = get_saved_papers(user_id)

    st.title("Saved Papers")

    if not saved:
        st.info("No saved papers yet. Use the Save button on your daily feed.")
        return

    st.caption(f"{len(saved)} paper{'s' if len(saved) != 1 else ''} saved")

    for item in saved:
        arxiv_id = item["arxiv_id"]
        date_str = item["created_at"][:10] if item.get("created_at") else ""
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        with st.container(border=True):
            st.markdown(
                f'<p style="font-size:1.05rem; font-weight:500; '
                f'margin-bottom:0.15rem;">{arxiv_id}</p>',
                unsafe_allow_html=True,
            )
            st.caption(f"Saved on {date_str}")
            st.markdown(f"[ArXiv page]({abs_url}) | [PDF]({pdf_url})")
