"""Onboarding page: topic & concept selection, free-text interests,
optional Scholar profile, and diversity slider.
"""

from __future__ import annotations

import streamlit as st

from pipeline.concept_tags import BROAD_CONCEPT_KEYS, CONCEPT_TAG_MAP
from pipeline.embed import EmbeddingModel
from pipeline.index import PaperIndex
from pipeline.interest_expander import embed_free_text_interests
from pipeline.scholar_parser import load_scholar_papers
from ui.components import TOPIC_LABELS, concept_tag_selector, free_text_input, topic_selector
from user.db import create_user
from user.profile import (
    SeedSignal,
    init_user_profile_v2,
    make_category_seed,
    make_concept_seed,
    make_freetext_seed,
    make_scholar_seed,
)


@st.cache_resource
def _get_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    st.title("ArXiv Daily")
    st.write("Personalized paper recommendations from arXiv, delivered daily.")
    st.divider()

    name = st.text_input("Your name", placeholder="Enter your display name")

    # -- arXiv category selection --
    st.write("**Pick arXiv categories you follow:**")
    selected_categories = topic_selector(index.category_centroids)

    # -- Concept tag selection --
    concept_embeddings = index.concept_embeddings or {}
    st.write("**Or pick some interdisciplinary themes:**")
    if index.concept_embeddings is None:
        st.warning(
            "Concept themes are unavailable. Run "
            "`python scripts/build_concept_embeddings.py` to enable them."
        )
    selected_concepts = concept_tag_selector(concept_embeddings)

    # -- Free-text interests --
    free_texts = free_text_input()

    # -- Optional Scholar profile --
    st.write("**Optional:** paste your Google Scholar profile URL for "
             "more precise first-day recommendations.")
    scholar_url = st.text_input(
        "Google Scholar URL",
        placeholder="https://scholar.google.com/citations?user=...",
    )

    # -- Diversity slider --
    st.write("**How broad should your daily papers be?**")
    diversity = st.slider(
        "Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = focused on your strongest interest · 1 = explore broadly",
    )

    if st.button("Start reading", type="primary"):
        if not name.strip():
            st.error("Please enter your name.")
            return
        if (
            not selected_categories
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

        # arXiv categories
        for code in selected_categories:
            label = next((l for l, c in TOPIC_LABELS.items() if c == code), code)
            seeds.append(make_category_seed(code, label, index.category_centroids[code]))

        # Concept tags
        for key in selected_concepts:
            tag = CONCEPT_TAG_MAP[key]
            broad = key in BROAD_CONCEPT_KEYS
            seeds.append(make_concept_seed(key, tag.label, concept_embeddings[key], broad=broad))

        # Free-text interests
        if free_texts:
            with st.spinner("Embedding your interests..."):
                model = _get_embed_model()
                for phrase, emb in embed_free_text_interests(free_texts, model):
                    seeds.append(make_freetext_seed(phrase, emb))

        # Scholar papers
        papers = None
        if scholar_url.strip():
            with st.spinner("Fetching your Scholar profile..."):
                papers = load_scholar_papers(scholar_url.strip())
            if papers:
                with st.spinner("Embedding your papers..."):
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
        user_id = create_user(
            name.strip(), centroids, k_u, diversity,
            thread_weights=result.thread_weights,
            thread_labels=result.thread_labels,
        )

        st.session_state["user_id"] = user_id
        st.session_state["user_centroids"] = centroids
        st.session_state["user_k_u"] = k_u
        st.session_state["user_diversity"] = diversity
        st.session_state["thread_labels"] = result.thread_labels
        st.session_state["thread_weights"] = result.thread_weights
        st.session_state["seed_thread_labels"] = result.seed_labels
        st.session_state["onboarded"] = True
        st.rerun()
