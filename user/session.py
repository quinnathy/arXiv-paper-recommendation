"""Streamlit session state helpers bridging st.session_state and the database.

Manages the lifecycle of user state within a Streamlit session:
- Initializing default session values for new visitors.
- Loading an existing user from the DB into the session.
- Syncing centroid updates between session state and DB.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from user.db import authenticate_user, get_user

TRANSIENT_USER_STATE_KEYS = (
    "active_arxiv_id",
    "active_tab",
    "active_tab_value",
    "learning_workspace",
    "overlay_page",
    "requested_tab",
    "responded",
    "right_sidebar_collapsed",
    "shown_ids",
    "todays_recs",
)
TRANSIENT_USER_STATE_PREFIXES = (
    "query_search_",
    "workspace_",
)


def _clear_transient_user_state() -> None:
    for key in TRANSIENT_USER_STATE_KEYS:
        st.session_state.pop(key, None)

    for key in list(st.session_state):
        if key.startswith(TRANSIENT_USER_STATE_PREFIXES):
            st.session_state.pop(key, None)


def load_or_init_session(db_path: str) -> None:
    """Initialize Streamlit session state with default user values.

    Called once at app startup. If user state is not already present in
    st.session_state, sets the following keys:
        - "user_id": None
        - "user_centroids": None
        - "user_k_u": None
        - "user_diversity": None
        - "onboarded": False

    Args:
        db_path: Path to the SQLite database (stored for later use).
    """
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
        st.session_state["user_centroids"] = None
        st.session_state["user_k_u"] = None
        st.session_state["user_diversity"] = None
        st.session_state["thread_labels"] = None
        st.session_state["thread_weights"] = None
        st.session_state["seed_thread_labels"] = None
        st.session_state["onboarded"] = False


def login_user(user_id: str, db_path: str) -> bool:
    """Load an existing user from the database into session state.

    Args:
        user_id: The UUID string of the user to load.
        db_path: Path to the SQLite database.

    Returns:
        True if the user was found and loaded, False otherwise.
    """
    user = get_user(user_id)
    if user is None:
        return False

    st.session_state["user_id"] = user["user_id"]
    st.session_state["user_centroids"] = user["centroids"]
    st.session_state["user_k_u"] = user["k_u"]
    st.session_state["user_diversity"] = user["diversity"]
    st.session_state["thread_labels"] = user.get("thread_labels")
    st.session_state["thread_weights"] = user.get("thread_weights")
    st.session_state["seed_thread_labels"] = None
    st.session_state["onboarded"] = True
    _clear_transient_user_state()
    return True


def login_with_credentials(username: str, password: str) -> bool:
    """Authenticate with username/password and load user into session."""
    user = authenticate_user(username, password)
    if user is None:
        return False

    st.session_state["user_id"] = user["user_id"]
    st.session_state["user_centroids"] = user["centroids"]
    st.session_state["user_k_u"] = user["k_u"]
    st.session_state["user_diversity"] = user["diversity"]
    st.session_state["thread_labels"] = user.get("thread_labels")
    st.session_state["thread_weights"] = user.get("thread_weights")
    st.session_state["seed_thread_labels"] = None
    st.session_state["onboarded"] = True
    _clear_transient_user_state()
    return True


def logout_user() -> None:
    """Clear user/auth session state for a clean logged-out state."""
    _clear_transient_user_state()
    for key in (
        "user_id",
        "user_centroids",
        "user_k_u",
        "user_diversity",
        "thread_labels",
        "thread_weights",
        "seed_thread_labels",
        "onboarded",
    ):
        st.session_state.pop(key, None)
    load_or_init_session("")


def save_centroids_to_session(centroids: np.ndarray) -> None:
    """Update the user centroids in the current Streamlit session state.

    Args:
        centroids: The new centroids, shape (k_u, 768), float32, unit-norm rows.
    """
    st.session_state["user_centroids"] = centroids


def is_onboarded() -> bool:
    """Check whether the current session user has completed onboarding.

    Returns:
        True if st.session_state["onboarded"] is True, False otherwise.
    """
    return st.session_state.get("onboarded", False)
