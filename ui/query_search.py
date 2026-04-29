"""Query-search UI for the main Daily Feed page."""

from __future__ import annotations

import streamlit as st

from pipeline.embed import EmbeddingModel
from pipeline.index import PaperIndex
from recommender.query_search import expand_query, search_papers
from user.db import get_seen_ids, log_feedback, update_centroids
from user.profile import apply_feedback
from user.session import save_centroids_to_session
from ui.components import paper_card


@st.cache_resource
def _get_query_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def _handle_search_feedback(arxiv_id: str, signal: str, index: PaperIndex) -> None:
    user_id = st.session_state["user_id"]
    centroids = st.session_state["user_centroids"]
    search_results = st.session_state.get("query_search_results", [])
    meta = next((r for r in search_results if r["id"] == arxiv_id), None)
    cluster_id = meta["cluster_id"] if meta else 0
    score = meta.get("search_score", 0.0) if meta else 0.0

    paper_idx = None
    for i, paper_meta in enumerate(index.paper_meta):
        if paper_meta["id"] == arxiv_id:
            paper_idx = i
            break

    log_feedback(user_id, arxiv_id, signal, cluster_id, score)

    if paper_idx is not None:
        paper_emb = index.embeddings[paper_idx]
        new_centroids = apply_feedback(centroids, paper_emb, signal)
        update_centroids(user_id, new_centroids)
        save_centroids_to_session(new_centroids)

    if "responded" not in st.session_state:
        st.session_state["responded"] = set()
    st.session_state["responded"].add(arxiv_id)
    st.rerun()


def render_query_search(index: PaperIndex) -> None:
    user_id = st.session_state["user_id"]

    with st.form("query_search_form"):
        query = st.text_input(
            "Search papers",
            placeholder="healthcare AI, medical image segmentation, LoRA...",
        )
        time_filter_label = st.selectbox(
            "Time range",
            options=["All time", "Past year", "Past 6 months", "Past 30 days"],
            index=0,
        )
        submitted = st.form_submit_button("Search")

    if submitted and query.strip():
        time_filter_days = {
            "All time": None,
            "Past year": 365,
            "Past 6 months": 183,
            "Past 30 days": 30,
        }[time_filter_label]
        expanded = expand_query(query)
        with st.spinner("Searching personalized paper results..."):
            model = _get_query_embed_model()
            query_embedding = model.embed_query(expanded)
            results = search_papers(
                query=query,
                query_embedding=query_embedding,
                user_centroids=st.session_state["user_centroids"],
                index=index,
                seen_ids=get_seen_ids(user_id),
                diversity=st.session_state["user_diversity"],
                n=20,
                time_filter_days=time_filter_days,
            )
        st.session_state["query_search_query"] = query
        st.session_state["query_search_expanded_query"] = expanded
        st.session_state["query_search_results"] = results

    results = st.session_state.get("query_search_results", [])
    if not results:
        return

    st.subheader(f"Search results for \"{st.session_state.get('query_search_query', '')}\"")
    for meta in results:
        paper_card(
            meta,
            on_like=lambda aid: _handle_search_feedback(aid, "like", index),
            on_save=lambda aid: _handle_search_feedback(aid, "save", index),
            on_skip=lambda aid: _handle_search_feedback(aid, "skip", index),
        )
        if st.button("Open in Research Lab", key=f"open_research_{meta['id']}"):
            st.session_state["active_arxiv_id"] = meta["id"]
            st.session_state["requested_tab"] = "Research Lab"
            st.rerun()
