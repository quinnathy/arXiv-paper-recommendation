"""Daily feed page for onboarded users.

Main page showing 5 recommended papers per day with like/save/skip actions.
Handles feedback processing: logging to DB, updating user centroids via EMA,
and persisting the updated centroids.
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st

from pipeline.index import PaperIndex
from recommender.retrieve import find_nearest_clusters
from recommender.engine import recommend
from recommender.visualization import build_cluster_dataframe, make_user_cluster_plot
from user.db import get_user, get_seen_ids, log_feedback, update_centroids
from user.profile import apply_feedback
from user.session import save_centroids_to_session
from ui.components import paper_card


VIZ_ARTIFACTS = (
    {
        "name": "UMAP",
        "coords": Path("data/diagnostics/umap_cluster_viz_coords.npy"),
        "indices": Path("data/diagnostics/umap_cluster_viz_indices.npy"),
        "metadata": Path("data/diagnostics/umap_cluster_viz_metadata.json"),
    },
    {
        "name": "PCA",
        "coords": Path("data/diagnostics/pca_cluster_viz_coords.npy"),
        "indices": Path("data/diagnostics/pca_cluster_viz_indices.npy"),
        "metadata": Path("data/diagnostics/pca_cluster_viz_metadata.json"),
    },
)


def _handle_feedback(
    arxiv_id: str, signal: str, index: PaperIndex
) -> None:
    """Process a feedback event: log, update centroids, mark responded."""
    user_id = st.session_state["user_id"]
    centroids = st.session_state["user_centroids"]

    recs = st.session_state.get("todays_recs", [])
    meta = next((r for r in recs if r["id"] == arxiv_id), None)
    cluster_id = meta["cluster_id"] if meta else 0
    score = meta.get("rec_score", 0.0) if meta else 0.0

    paper_idx = None
    for i, pm in enumerate(index.paper_meta):
        if pm["id"] == arxiv_id:
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


@st.cache_data(show_spinner=False)
def _load_embedding_viz_dataframe(
    artifact_name: str,
    coords_path: str,
    indices_path: str,
) -> tuple[str, dict, np.ndarray, np.ndarray]:
    coords = np.load(coords_path)
    indices = np.load(indices_path)
    with open(next(a["metadata"] for a in VIZ_ARTIFACTS if a["name"] == artifact_name), "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return artifact_name, metadata, coords, indices


def _available_viz_artifact() -> dict | None:
    for artifact in VIZ_ARTIFACTS:
        if artifact["coords"].exists() and artifact["indices"].exists() and artifact["metadata"].exists():
            return artifact
    return None


def _project_user_centroids_if_possible(artifact: dict, centroids: np.ndarray) -> np.ndarray | None:
    if artifact["name"] != "UMAP":
        return None
    try:
        with open("data/diagnostics/umap_cluster_viz_model.pkl", "rb") as fh:
            model = pickle.load(fh)
        return model.transform(centroids).astype(np.float32)
    except Exception:
        return None


def _render_embedding_space(index: PaperIndex, recs: list[dict]) -> None:
    artifact = _available_viz_artifact()
    if artifact is None:
        st.info(
            "Embedding visualization artifacts are not available yet. "
            "Run `python scripts/diagnostics/visualize_clusters_umap.py` "
            "or `python scripts/diagnostics/visualize_clusters_pca.py`."
        )
        return

    try:
        artifact_name, viz_meta, coords, indices = _load_embedding_viz_dataframe(
            artifact["name"],
            str(artifact["coords"]),
            str(artifact["indices"]),
        )
        sampled_meta = [index.paper_meta[int(i)] for i in indices]
        df = build_cluster_dataframe(coords, indices, sampled_meta, cluster_ids=index.cluster_ids)
    except Exception as exc:
        st.warning(f"Could not load embedding visualization artifacts: {exc}")
        return

    color_by = st.selectbox("Color by", ["primary_category", "top_level_category", "cluster"])
    categories = sorted(df["primary_category"].dropna().unique().tolist())
    selected_categories = st.multiselect("Filter categories", categories)
    clusters = sorted(int(c) for c in df["cluster"].dropna().unique().tolist())
    selected_clusters = st.multiselect("Filter clusters", clusters)
    if len(df) <= 500:
        max_points = len(df)
    else:
        max_points = st.slider("Max points", 500, len(df), min(10_000, len(df)), step=500)
    show_background = st.checkbox("Show background sample", True)
    show_searched = st.checkbox("Show searched clusters", True)
    show_served = st.checkbox("Show served papers", True)
    show_centroids = st.checkbox("Show user centroids", True)

    filtered = df
    if selected_categories:
        filtered = filtered[filtered["primary_category"].isin(selected_categories)]
    if selected_clusters:
        filtered = filtered[filtered["cluster"].isin(selected_clusters)]
    if len(filtered) > max_points:
        filtered = filtered.sample(n=max_points, random_state=42)

    user_centroids = st.session_state.get("user_centroids")
    searched_clusters = None
    if show_searched and user_centroids is not None:
        searched_clusters = find_nearest_clusters(
            user_centroids,
            index.centroids,
            diversity=st.session_state.get("user_diversity", 0.5),
        )

    id_to_index = {meta["id"]: i for i, meta in enumerate(index.paper_meta)}
    served_indices = None
    if show_served:
        served_indices = [id_to_index[r["id"]] for r in recs if r.get("id") in id_to_index]

    centroid_coords = None
    if show_centroids and user_centroids is not None:
        centroid_coords = _project_user_centroids_if_possible(artifact, user_centroids)

    if not show_background:
        keep_indices: set[int] = set(served_indices or [])
        if searched_clusters:
            keep_indices.update(
                filtered.loc[filtered["cluster"].isin(searched_clusters), "paper_index"].astype(int).tolist()
            )
        filtered = filtered[filtered["paper_index"].isin(keep_indices)]

    if filtered.empty:
        st.info("No sampled points match the current visualization filters.")
        return

    try:
        fig = make_user_cluster_plot(
            filtered,
            user_centroid_coords=centroid_coords,
            searched_cluster_ids=searched_clusters if show_searched else None,
            served_paper_indices=served_indices if show_served else None,
            color_by=color_by,
        )
        fig.update_layout(title=f"{artifact_name} Paper Embedding Space")
        st.plotly_chart(fig, use_container_width=True)
        if show_centroids and centroid_coords is None:
            st.caption("User centroids can be overlaid when UMAP artifacts include a fitted transformable model.")
        if viz_meta.get("explained_variance_ratio"):
            ratios = viz_meta["explained_variance_ratio"]
            st.caption(f"PCA explained variance: PC1 {ratios[0] * 100:.2f}%, PC2 {ratios[1] * 100:.2f}%.")
    except Exception as exc:
        st.warning(f"Could not render embedding visualization: {exc}")


def render_daily_feed(index: PaperIndex, db_path: str) -> None:
    """Render the daily paper feed for an onboarded user.

    Args:
        index: The loaded PaperIndex for recommendation lookups.
        db_path: Path to the SQLite database for feedback logging.
    """
    user_id = st.session_state["user_id"]
    user = get_user(user_id)

    st.title(f"Good morning, {user['display_name']}")
    st.caption(date.today().strftime("%A, %B %d, %Y"))

    if "shown_ids" not in st.session_state:
        st.session_state["shown_ids"] = set()

    if "todays_recs" not in st.session_state:
        centroids = st.session_state["user_centroids"]
        diversity = st.session_state["user_diversity"]
        seen_ids = get_seen_ids(user_id)
        excluded_ids = seen_ids | st.session_state["shown_ids"]
        with st.spinner("Finding your papers..."):
            recs = recommend(centroids, excluded_ids, index, diversity=diversity, n=5)
        st.session_state["todays_recs"] = recs
        st.session_state["shown_ids"].update(r["id"] for r in recs)

    if "responded" not in st.session_state:
        st.session_state["responded"] = set()

    recs = st.session_state["todays_recs"]

    if not recs:
        st.info("You've explored all available papers in your areas for this session.")
        if st.button("Reset and start fresh"):
            st.session_state["shown_ids"] = set()
            st.session_state.pop("todays_recs", None)
            st.session_state.pop("responded", None)
            st.rerun()
        return
    if len(recs) < 5:
        st.warning("You've seen most papers in your areas. Here's what we found:")

    st.subheader(f"Your {len(recs)} paper{'s' if len(recs) != 1 else ''} for today")

    for meta in recs:
        paper_card(
            meta,
            on_like=lambda aid: _handle_feedback(aid, "like", index),
            on_save=lambda aid: _handle_feedback(aid, "save", index),
            on_skip=lambda aid: _handle_feedback(aid, "skip", index),
        )

    with st.expander("Embedding space", expanded=False):
        _render_embedding_space(index, recs)

    responded = st.session_state.get("responded", set())
    rec_ids = {r["id"] for r in recs}
    if rec_ids and rec_ids.issubset(responded):
        st.success("Come back tomorrow for new recommendations!")

    # Demo button: simulate next day's digest
    st.divider()
    if st.button("Recommend again (demo)", help="Fetch a fresh batch of 5 papers to evaluate quality"):
        st.session_state.pop("todays_recs", None)
        st.session_state.pop("responded", None)
        st.rerun()
