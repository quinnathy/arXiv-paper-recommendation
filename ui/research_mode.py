import streamlit as st
from streamlit.components.v1 import iframe
from user.db import save_research_note, get_all_notes
from fpdf import FPDF
from pipeline.transcribe import get_paper_markdown

def render_research_mode():
    user_id = st.session_state["user_id"]
    active_id = st.session_state.get("active_arxiv_id", "No Paper Selected")

    st.title("🔬 Research Lab")
    
    col_source, col_wall = st.columns([0.5, 0.5])
    
    with col_source:
        st.subheader(f"Current Paper: {active_id}")
        if active_id != "No Paper Selected":
            pdf_url = f"https://arxiv.org/pdf/{active_id}.pdf"
            iframe(pdf_url, height=800)
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
