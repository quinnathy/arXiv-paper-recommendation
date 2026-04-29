"""ArXiv Daily — Streamlit entry point.

Thin shell that initializes the database, loads the paper index (cached),
and routes to the appropriate page based on onboarding state.
"""

from pipeline.runtime import configure_single_thread_runtime

configure_single_thread_runtime()

from pathlib import Path

import streamlit as st

from pipeline.index import PaperIndex
from user.db import init_db
from user.session import load_or_init_session, is_onboarded, logout_user
from ui.onboarding import render_onboarding
from ui.daily_feed import render_daily_feed
from ui.research_mode import render_research_mode
from ui.profile_page import render_profile_page
from ui.archive_page import render_archive_page
from user.db import init_db, get_user, get_saved_papers 
from ui.learning_mode import render_learning_mode, render_workspace_sidebar

DB_PATH = "data/arxiv_rec.db"

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
# -- route -----------------------------------------------------------------
if not is_onboarded():
    render_onboarding(index, DB_PATH)
else:
    # 1. --- LEFT SIDEBAR (Native Navigation) ---
    with st.sidebar:
        user_id = st.session_state["user_id"]
        user_data = get_user(user_id) #

        # Profile Image and Name
        st.image("https://www.gravatar.com/avatar/0000?d=mp&f=y", width=60)
        st.markdown(f"**{user_data['display_name']}**")
        
        if st.button("👤 User Profile", use_container_width=True):
            st.session_state["overlay_page"] = "profile"
        
        st.markdown(
            """<style>
            /* Sidebar nav buttons — bookmark style */
            section[data-testid='stSidebar'] button[kind='secondary'] {
                font-size: 1rem !important;
                font-weight: 600 !important;
                padding: 0.6rem 1rem !important;
                border-radius: 0.5rem !important;
                text-align: left !important;
                justify-content: flex-start !important;
            }
            /* Log-out button — outlined / distinct */
            section[data-testid='stSidebar'] button[kind='secondary'].logout-btn,
            section[data-testid='stSidebar'] div[data-testid='stButton']:last-child button {
                font-weight: 400 !important;
                border: 1px solid rgba(150,150,150,0.4) !important;
                background: transparent !important;
                color: inherit !important;
                font-size: 0.9rem !important;
                padding: 0.4rem 1rem !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        profile_clicked = st.button("User Profile", use_container_width=True)
        st.divider()

        if st.button("🔍 Explore Mode", use_container_width=True):
            st.session_state["active_tab"] = "Daily Feed"
            st.session_state.pop("overlay_page", None)
            
        if st.button("📂 Archive", use_container_width=True):
            st.session_state["overlay_page"] = "archive"

        # Spacer replacement for st.spacer
        st.markdown("<br>" * 5, unsafe_allow_html=True) 
        if st.button("🚪 Log out", use_container_width=True):
            logout_user()
            st.rerun()

    # --- Mode pills (main content area) ---
    st.markdown(
        """<style>
        div[data-testid='stPills'] div[role='tablist'] button {
            font-size: 1.05rem !important;
            padding: 0.45rem 1.6rem !important;
            font-weight: 600 !important;
            border-radius: 2rem !important;
        }
        div[data-testid='stPills'] div[role='tablist'] button[aria-checked='true'] {
            font-weight: 700 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    if "active_tab_value" not in st.session_state:
        st.session_state["active_tab_value"] = st.session_state.pop(
            "active_tab", "Daily Feed"
        )

    if st.session_state["active_tab_value"] == "Learning Mode":
        st.session_state["active_tab_value"] = "Workspace"

    if "requested_tab" in st.session_state:
        st.session_state["active_tab_value"] = st.session_state.pop("requested_tab")
        st.session_state.pop("active_tab_widget", None)

    if st.session_state["active_tab_value"] == "Learning Mode":
        st.session_state["active_tab_value"] = "Workspace"

    active_tab = st.pills(
        "mode",
        options=["Daily Feed", "Workspace", "Research Lab"],
        default=st.session_state["active_tab_value"],
        key="active_tab_widget",
        label_visibility="collapsed",
    )
    st.session_state["active_tab_value"] = active_tab

    render_workspace_sidebar(index, active_tab)

    # Persist overlay selection across reruns
    if profile_clicked:
        st.session_state["overlay_page"] = "profile"

    overlay = st.session_state.get("overlay_page")
    if overlay == "profile":
        if st.button("← Back"):
            st.session_state.pop("overlay_page", None)
            st.rerun()
        render_profile_page()
    else:
        if active_tab == "Research Lab":
            render_research_mode()
        elif active_tab == "Workspace":
            render_learning_mode(index)
        else:
            render_daily_feed(index, DB_PATH)