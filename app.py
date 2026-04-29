"""ArXiv Daily — Streamlit entry point.

Thin shell that initializes the database, loads the paper index (cached),
and routes to the appropriate page based on onboarding state.
"""

from pipeline.runtime import configure_single_thread_runtime

configure_single_thread_runtime()

from pathlib import Path

import streamlit as st

from pipeline.index import PaperIndex
from user.db import get_user, init_db
from user.session import load_or_init_session, is_onboarded, logout_user
from ui.onboarding import render_onboarding
from ui.daily_feed import render_daily_feed
from ui.research_mode import render_research_mode
from ui.profile_page import render_profile_page
from ui.archive_page import render_archive_page
from ui.learning_mode import render_learning_mode, render_workspace_sidebar

DB_PATH = "data/arxiv_rec.db"


def _clear_query_search_state() -> None:
    for key in (
        "query_search_input",
        "query_search_time_filter",
        "query_search_query",
        "query_search_expanded_query",
        "query_search_results",
    ):
        st.session_state.pop(key, None)

st.set_page_config(
    page_title="ArXiv Daily",
    page_icon="📄",
    layout="centered",
)

# -- global stylesheet -----------------------------------------------------
_css_path = Path(__file__).parent / "ui" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_index() -> PaperIndex:
    """Load the paper index once and cache it across Streamlit reruns.

    The ~6 GB embedding matrix must be loaded exactly once per process.
    st.cache_resource ensures this singleton behavior.

    Returns:
        A loaded PaperIndex instance.
    """
    idx = PaperIndex()
    idx.load()
    return idx


# -- init ------------------------------------------------------------------
init_db(DB_PATH)
load_or_init_session(DB_PATH)
index = load_index()

# -- check if pipeline has run ---------------------------------------------
if not index.is_loaded():
    st.error(
        "Data not found. Run: python scripts/run_offline_pipeline.py --limit 50000"
    )
    st.stop()

# -- route -----------------------------------------------------------------
if not is_onboarded():
    render_onboarding(index, DB_PATH)
else:
    # 1. --- LEFT SIDEBAR (Native Navigation) ---
    with st.sidebar:
        user_id = st.session_state["user_id"]
        user_data = get_user(user_id)

        # Profile Image and Name
        st.image("https://www.gravatar.com/avatar/0000?d=mp&f=y", width=60)
        st.markdown(f"**{user_data['display_name']}**")

        if st.button("User Profile", use_container_width=True):
            st.session_state["overlay_page"] = "profile"

        st.divider()

        if st.button("Explore Mode", use_container_width=True):
            st.session_state["active_tab_value"] = "Daily Feed"
            st.session_state.pop("active_tab_widget", None)
            st.session_state.pop("overlay_page", None)
            _clear_query_search_state()

        if st.button("Archive", use_container_width=True):
            st.session_state["overlay_page"] = "archive"

        # Spacer replacement
        st.markdown("<br>" * 5, unsafe_allow_html=True)
        
        if st.button("Log out", use_container_width=True):
            logout_user()
            st.rerun()

    # 2. --- MAIN CONTENT AREA ---
    # Syncing tab state
    if "active_tab_value" not in st.session_state:
        st.session_state["active_tab_value"] = st.session_state.pop("active_tab", "Daily Feed")

    if "requested_tab" in st.session_state:
        st.session_state["active_tab_value"] = st.session_state.pop("requested_tab")
        st.session_state.pop("active_tab_widget", None)

    # Mode pills
    active_tab = st.pills(
        "mode",
        options=["Daily Feed", "Workspace", "Research Lab"],
        default=st.session_state["active_tab_value"],
        key="active_tab_widget",
        label_visibility="collapsed",
    )
    st.session_state["active_tab_value"] = active_tab

    # Right Sidebar (Folders/Files)
    render_workspace_sidebar(index, active_tab)

    # 3. --- ROUTING ---
    overlay = st.session_state.get("overlay_page")
    
    if overlay == "profile":
        if st.button("← Back to Feed"):
            st.session_state.pop("overlay_page", None)
            st.rerun()
        render_profile_page(index)
    
    elif overlay == "archive":
        if st.button("← Back to Feed"):
            st.session_state.pop("overlay_page", None)
            st.rerun()
        render_archive_page()
        
    else:
        # Standard Mode Routing
        if active_tab == "Research Lab":
            render_research_mode(index)
        elif active_tab == "Workspace":
            render_learning_mode(index)
        else:
            render_daily_feed(index, DB_PATH)
