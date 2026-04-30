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


QUERY_SEARCH_EXAMPLES = [
    "retrieval-augmented generation for factual and citation-grounded LLMs",
    "diffusion models for medical image reconstruction with limited labels",
    "graph neural networks for molecular property prediction and drug discovery",
    "reinforcement learning for safe robotic navigation in uncertain environments",
    "neural operators for weather forecasting and climate downscaling",
    "multimodal learning across text, image, audio, and video",
    "privacy-preserving federated learning for distributed healthcare data",
    "interpretable machine learning for genomics and single-cell data",
    "AI alignment methods for scalable oversight and reward modeling",
    "time series forecasting for financial and macroeconomic data",
    "representation learning for neural population activity and behavior",
    "efficient transformers for sparse attention and long-context inference",
    "causal inference for treatment effects in observational healthcare data",
    "formal verification and program synthesis for code correctness",
    "autonomous driving perception and planning under uncertainty",
    "recommender systems balancing personalization, diversity, and fairness",
    "variational quantum algorithms and quantum machine learning",
    "speech recognition for noisy multilingual low-resource settings",
    "optimization methods for sharpness-aware and generalizable deep learning",
]


def _rotating_search_placeholder() -> str:
    idx = st.session_state.get("query_search_example_idx", 0)
    st.session_state["query_search_example_idx"] = idx + 1
    return QUERY_SEARCH_EXAMPLES[idx % len(QUERY_SEARCH_EXAMPLES)]


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

    st.markdown("**Search papers by topic, method, dataset, or research question...**")

    search_col, options_col = st.columns([0.92, 0.08])
    with search_col:
        query = st.text_input(
            "Search papers by topic, method, dataset, or research question...",
            placeholder=_rotating_search_placeholder(),
            key="query_search_input",
            label_visibility="collapsed",
        )
    with options_col:
        options_open = st.session_state.get("query_search_options_open", False)
        if st.button(
            "▲" if options_open else "▼",
            key="query_search_options_toggle",
            help="Search options",
            use_container_width=True,
        ):
            st.session_state["query_search_options_open"] = not options_open
            st.rerun()

    time_filter_label = st.session_state.get("query_search_time_filter", "All time")
    if st.session_state.get("query_search_options_open", False):
        time_filter_label = st.selectbox(
            "Time range",
            options=["All time", "Past year", "Past 6 months", "Past 30 days"],
            index=0,
            key="query_search_time_filter",
        )
    submitted = st.button("Search", key="query_search_submit")

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
