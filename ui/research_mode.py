import streamlit as st
from pipeline.embed import EmbeddingModel
from pipeline.index import PaperIndex
from recommender.query_search import expand_query, search_papers
from user.db import (
    get_all_notes,
    get_seen_ids,
    log_feedback,
    save_research_note,
    update_centroids,
)
from user.profile import apply_feedback
from user.session import save_centroids_to_session
from ui.components import paper_card
from pipeline.transcribe import get_paper_markdown


@st.cache_resource
def _get_query_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def _handle_search_feedback(arxiv_id: str, signal: str, index: PaperIndex) -> None:
    user_id = st.session_state["user_id"]
    centroids = st.session_state["user_centroids"]
    search_results = st.session_state.get("research_search_results", [])
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


def _render_query_search(index: PaperIndex) -> None:
    user_id = st.session_state["user_id"]

    with st.form("research_query_search_form"):
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
        st.session_state["research_search_query"] = query
        st.session_state["research_search_expanded_query"] = expanded
        st.session_state["research_search_results"] = results

    results = st.session_state.get("research_search_results", [])
    if not results:
        return

    st.subheader(f"Search results for “{st.session_state.get('research_search_query', '')}”")
    for meta in results:
        paper_card(
            meta,
            on_like=lambda aid: _handle_search_feedback(aid, "like", index),
            on_save=lambda aid: _handle_search_feedback(aid, "save", index),
            on_skip=lambda aid: _handle_search_feedback(aid, "skip", index),
        )
        if st.button("Open in Research Lab", key=f"open_research_{meta['id']}"):
            st.session_state["active_arxiv_id"] = meta["id"]
            st.rerun()


def render_research_mode(index: PaperIndex):
    user_id = st.session_state["user_id"]
    active_id = st.session_state.get("active_arxiv_id", "No Paper Selected")

    st.title("🔬 Research Lab")
    _render_query_search(index)
    st.divider()
    
    col_source, col_wall = st.columns([0.5, 0.5])
    
    with col_source:
        st.subheader(f"Current Paper: {active_id}")
        if active_id != "No Paper Selected":
            pdf_url = f"https://arxiv.org/pdf/{active_id}.pdf"
            st.components.v1.iframe(pdf_url, height=800)
        else:
            st.info("Select a paper from your feed to start researching.")
        if st.button("✨ Transcribe with AI"):
            # Set use_mock=False when you're ready for the real thing!
            md = get_paper_markdown(active_id, use_mock=True)
            st.session_state["transcription"] = md

        if "transcription" in st.session_state:
            st.text_area("Select and copy from here:", st.session_state["transcription"], height=600)

    with col_wall:
        st.subheader("📓 Infinite Wall")
        
        # Note Input
        with st.form("new_note_form", clear_on_submit=True):
            note_content = st.text_area("Snippet/Brainstorm", placeholder="Paste Marker output here...")
            submitted = st.form_submit_button("Append to Wall")
            if submitted and note_content:
                save_research_note(user_id, note_content, active_id)
                st.rerun()

        # Display the Wall
        notes = get_all_notes(user_id)
        for content, aid, ts in reversed(notes): # Show newest at top for the wall
            with st.container(border=True):
                st.markdown(content)
                st.caption(f"Source: {aid} | {ts[:10]}")

        # PDF Export Trigger
        if notes:
            st.divider()
            generate_pdf_download(notes)

def generate_pdf_download(notes):
    """The PDF Export Tool logic."""
    try:
        from fpdf import FPDF
    except ModuleNotFoundError:
        st.warning("PDF export requires `fpdf2`. Run: pip install -r requirements.txt")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ArXiv Daily: Research Export", ln=True, align='C')
    pdf.ln(10)

    for content, aid, ts in notes:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128)
        pdf.cell(0, 5, f"Source: {aid} | Date: {ts[:10]}", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(0)
        pdf.multi_cell(0, 10, content)
        pdf.ln(5)

    pdf_output = pdf.output(dest='S')
    st.download_button(
        label="📥 Download Wall as PDF",
        data=pdf_output,
        file_name="research_notes.pdf",
        mime="application/pdf"
    )
