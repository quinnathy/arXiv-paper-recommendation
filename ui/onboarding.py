"""Onboarding page for new users.

Shown when the user has not yet completed the initial setup.
Collects a display name and topic preferences, then creates a user profile
with an initial embedding derived from the selected category centroids.
"""

from __future__ import annotations

import streamlit as st

from pipeline.index import PaperIndex
from user.db import create_user
from user.profile import init_embedding_from_topics
from ui.components import inject_design, topic_selector


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    """Render the onboarding page for a new (not yet onboarded) user.

    Args:
        index: The loaded PaperIndex (needed for category_centroids).
        db_path: Path to the SQLite database.
    """
    
    inject_design()
    
    # Header with Serif font (handled by CSS h1)
    st.markdown("""
        <h1>
            The Morning Briefing <br> for the Modern Researcher
        </h1>
    """, unsafe_allow_html=True)    
    # Description with custom line-height for readability
    st.markdown("""
        <p style="font-family: 'Newsreader', serif; line-height: 1.3; font-size: 1.1rem; font-weight:500; color: black;">
            <b>ArXiv Daily</b> is a personalized discovery engine that cuts through the noise 
            of over two million scientific publications. We map your research interests into 
            a high-dimensional geometric space, allowing us to find the <b>semantic fingerprint</b> 
            of the papers that matter to you most.
        </p>
    """, unsafe_allow_html=True)
    name = st.text_input("Welcome,", placeholder="Enter your display name")

    st.markdown("""
        <p style="font-family: 'Newsreader', serif; line-height: 1.3; font-size: 1.1rem; font-weight:500; color: black;">
            What are we interested in today? This will help us create your initial profile and tailor your daily paper recommendations.
        </p>
    """, unsafe_allow_html=True)
    selected_categories = topic_selector(index.category_centroids)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Start reading", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Please enter your name.")
            elif not selected_categories:
                st.error("Please select at least one topic.")
            else:
                embedding = init_embedding_from_topics(
                    selected_categories, index.category_centroids
                )
                user_id = create_user(name.strip(), embedding)

                st.session_state["user_id"] = user_id
                st.session_state["user_embedding"] = embedding
                st.session_state["onboarded"] = True
                st.rerun()
