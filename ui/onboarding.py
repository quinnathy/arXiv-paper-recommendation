"""Onboarding page: topic & concept selection, free-text interests,
optional Scholar profile, and diversity slider.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from pipeline.concept_tags import CONCEPT_TAG_MAP
from pipeline.embed import EmbeddingModel
from pipeline.index import PaperIndex
from pipeline.interest_expander import embed_free_text_interests
from pipeline.scholar_parser import load_scholar_papers
from ui.components import (
    MAX_ONBOARDING_TAGS,
    TOPIC_LABELS,
    expand_topic_labels,
    free_text_input,
    loading_spinner_with_message,
    unified_tag_selector,
)
from user.db import create_user
from user.session import login_with_credentials
from user.profile import (
    SeedSignal,
    init_user_profile_v2,
    make_category_seed,
    make_concept_seed,
    make_freetext_seed,
    make_scholar_seed,
)


@st.cache_resource(show_spinner=False)
def _get_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def make_category_seeds_from_topic_labels(
    selected_labels: list[str],
    category_centroids: dict[str, np.ndarray],
) -> list[SeedSignal]:
    """Build category seeds from human-readable onboarding labels."""
    topic_keys = expand_topic_labels(
        selected_labels=selected_labels,
        topic_labels=TOPIC_LABELS,
        category_centroids=category_centroids,
    )
    if selected_labels and not topic_keys:
        raise ValueError(
            "None of the selected onboarding topics are available in the "
            "current corpus category centroids."
        )

    seeds: list[SeedSignal] = []
    for code in topic_keys:
        label = next(
            (
                selected_label
                for selected_label in selected_labels
                if code in TOPIC_LABELS.get(selected_label, [])
            ),
            code,
        )
        seeds.append(make_category_seed(code, label, category_centroids[code]))
    return seeds


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    st.title("ArXiv Daily")
    st.write("Personalized paper recommendations from arXiv, delivered daily.")
    st.divider()
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = None

    mode = st.session_state["auth_mode"]
    if mode is None:
        st.subheader("Welcome")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Log in", width="stretch"):
                st.session_state["auth_mode"] = "Log in"
                st.rerun()
        with col2:
            if st.button("Sign up", width="stretch"):
                st.session_state["auth_mode"] = "Create account"
                st.rerun()
        with col3:
            if st.button("Continue as guest", width="stretch"):
                st.session_state["auth_mode"] = "Continue as guest"
                st.rerun()
        return

    if mode == "Log in":
        if st.button("Back"):
            st.session_state["auth_mode"] = None
            st.rerun()
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in", type="primary"):
            if not username.strip() or not password:
                st.error("Please enter both username and password.")
                return
            if not login_with_credentials(username, password):
                st.error("Invalid username or password.")
                return
            st.success("Welcome back!")
            st.rerun()
        return

    if st.button("Back"):
        st.session_state["auth_mode"] = None
        st.rerun()

    is_guest = mode == "Continue as guest"
    if not is_guest:
        username = st.text_input("Username", placeholder="Choose a username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        name = username.strip()
    else:
        st.caption("Guest mode: no account required. Your session is temporary.")
        username = ""
        password = ""
        confirm_password = ""
        name = st.text_input("Your name", placeholder="Enter your display name")

    st.write("")

    # -- Topic & concept tag selection (unified) --
    concept_embeddings = index.concept_embeddings or {}
    st.write("**Pick topics and themes you're interested in:**")
    if index.concept_embeddings is None:
        st.caption(
            "Run `python scripts/build_concept_embeddings.py` to unlock "
            "more themes."
        )
    selected_topic_labels, selected_concepts = unified_tag_selector(
        index.category_centroids, concept_embeddings,
    )

    st.write("")

    # -- Free-text interests --
    free_texts = free_text_input()

    st.write("")

    # -- Optional Scholar profile --
    st.write("**Paste your Google Scholar profile URL** (optional)")
    scholar_url = st.text_input(
        "Google Scholar URL",
        placeholder="https://scholar.google.com/citations?user=...",
        label_visibility="collapsed",
    )

    st.write("")

    # -- Diversity slider --
    st.write("**Exploration range**")
    st.caption(
        "Lower values surface papers closest to your core interests. "
        "Higher values mix in papers from neighboring fields for serendipity."
    )
    diversity = st.slider(
        "Exploration range",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        label_visibility="collapsed",
    )

    if st.button("Start reading", type="primary"):
        if not is_guest:
            if not username.strip():
                st.error("Please choose a username.")
                return
            if len(password) < 8:
                st.error("Password must be at least 8 characters.")
                return
            if password != confirm_password:
                st.error("Passwords do not match.")
                return
        if not is_guest and not username.strip():
            st.error("Please choose a username.")
            return
        if is_guest and not name.strip():
            st.error("Please enter your name.")
            return
        if len(selected_topic_labels) + len(selected_concepts) > MAX_ONBOARDING_TAGS:
            st.error(f"Please choose at most {MAX_ONBOARDING_TAGS} tags.")
            return
        if (
            not selected_topic_labels
            and not selected_concepts
            and not free_texts
            and not scholar_url.strip()
        ):
            st.error(
                "Please provide at least one category, theme, free-text "
                "interest, or Scholar profile."
            )
            return

        # -- Assemble seed signals --
        seeds: list[SeedSignal] = []

        # arXiv categories expanded from human-readable onboarding topics.
        try:
            seeds.extend(
                make_category_seeds_from_topic_labels(
                    selected_topic_labels,
                    index.category_centroids,
                )
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        # Concept tags
        for key in selected_concepts:
            tag = CONCEPT_TAG_MAP[key]
            seeds.append(
                make_concept_seed(
                    key,
                    tag.label,
                    concept_embeddings[key],
                )
            )

        # Free-text interests
        if free_texts:
            with loading_spinner_with_message():
                model = _get_embed_model()
                for phrase, emb in embed_free_text_interests(free_texts, model):
                    seeds.append(make_freetext_seed(phrase, emb))

        # Scholar papers
        papers = None
        if scholar_url.strip():
            with loading_spinner_with_message():
                papers = load_scholar_papers(scholar_url.strip())
            if papers:
                with loading_spinner_with_message():
                    model = _get_embed_model()
                    paper_embeddings = model.embed_papers(papers)
                    for i, paper in enumerate(papers):
                        seeds.append(make_scholar_seed(paper["title"], paper_embeddings[i]))
            else:
                st.warning("Could not load Scholar profile. "
                           "Continuing with other signals.")

        # -- Initialize profile --
        if not seeds:
            st.error(
                "Could not build any usable profile seeds. Please add another "
                "interest signal."
            )
            return

        result = init_user_profile_v2(seeds)
        centroids = result.centroids
        k_u = centroids.shape[0]
        try:
            user_id = create_user(
                (username.strip() if not is_guest else name.strip()),
                centroids,
                k_u,
                diversity,
                thread_weights=result.thread_weights,
                thread_labels=result.thread_labels,
                username=username.strip() if not is_guest else None,
                password=password if not is_guest else None,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.session_state["user_id"] = user_id
        st.session_state["user_centroids"] = centroids
        st.session_state["user_k_u"] = k_u
        st.session_state["user_diversity"] = diversity
        st.session_state["thread_labels"] = result.thread_labels
        st.session_state["thread_weights"] = result.thread_weights
        st.session_state["seed_thread_labels"] = result.seed_labels
        st.session_state["onboarded"] = True
        st.rerun()
