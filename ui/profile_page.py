"""User profile page showing account info and research stats."""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from user.db import get_feedback_counts, get_interacted_paper_count, get_user
from ui.daily_feed import _render_embedding_space


def render_profile_page(index) -> None:
    """Render the user profile page."""
    user_id = st.session_state["user_id"]
    user = get_user(user_id)
    counts = get_feedback_counts(user_id)
    interacted_count = get_interacted_paper_count(user_id)

    st.title(f"@{user['username']}'s Profile")

    # --- Identity card ---
    col_avatar, col_info = st.columns([0.15, 0.85])
    with col_avatar:
        # FIX: Changed 'user_data' to 'user' to match the variable assigned above
        avatar_file = user.get("profile_pic") or "default_avatar.jpg"
    
        # Define the path using assets/avatars to match your project structure [cite: 2402, 2427]
        avatar_path = Path(".streamlit/static/avatars") / avatar_file

        if avatar_path.exists():
            # Read the file as binary to ensure it renders correctly in any environment [cite: 2404, 2425]
            with open(avatar_path, "rb") as f:
                st.image(f.read(), width=150)
        else:
            # Fallback to a standard emoji icon if the file is missing on disk [cite: 2421, 2427]
            st.markdown(
                '<div style="font-size:3.5rem; text-align:center;">&#129489;</div>',
                unsafe_allow_html=True,
            )
            
    with col_info:
        st.markdown(f"### {user['display_name']}")
        if user.get("username"):
            st.caption(f"@{user['username']}")
        created = user["created_at"][:10] if user["created_at"] else ""
        last_active = user["last_active"][:10] if user.get("last_active") else ""
        st.caption(f"Member since {created}  &middot;  Last active {last_active}")

    st.divider()

    # --- Activity metrics ---
    st.markdown("**Activity**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Papers seen", interacted_count)
    m2.metric("Liked", counts.get("like", 0))
    m3.metric("Saved", counts.get("save", 0))
    m4.metric("Skipped", counts.get("skip", 0))

    st.divider()

    # --- Moving the Embedding Space Viz here ---
    st.subheader("Your Research Footprint")
    st.caption("This map shows where your interests lie relative to the ArXiv corpus.")
    
    # We pass an empty list for 'recs' since we are in profile mode, 
    # or you can pass st.session_state.get("todays_recs", [])
    _render_embedding_space(index, st.session_state.get("todays_recs", []))

    st.divider()

    # --- Settings snapshot ---
    st.markdown("**Preferences**")
    st.write(f"Exploration range: **{user['diversity']:.1f}**")