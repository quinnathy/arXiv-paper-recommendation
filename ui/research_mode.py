import streamlit as st
from PIL import Image
import io
import base64
import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from streamlit_cropper import st_cropper
except ImportError:
    st_cropper = None

from user.db import get_all_notes, save_research_note
from ui.components import loading_spinner_with_message
from datetime import datetime

st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
    }

    section[data-testid="stMainBlockContainer"] {
        padding-top: 0.75rem !important;
    }

    div.block-container {
        padding-top: 0.75rem !important;
    }

    h1:first-child {
        margin-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def render_research_mode(index=None):
    if fitz is None:
        st.error(
            "Research Lab requires `pymupdf`. Install it with "
            "`pip install pymupdf` or `pip install -r requirements.txt`."
        )
        return

    user_id = st.session_state["user_id"]
    active_id = st.session_state.get("active_arxiv_id", "No Paper Selected")

    st.title("Research Lab")
    
    if active_id == "No Paper Selected":
        st.info("Select a paper from your feed to start a session.")
        return

    # --- STEP 1: DOWNLOAD CHECK ---
    # We do this before setting up columns to avoid the FileNotFoundError
    try:
        from pipeline.transcribe import download_pdf

        with loading_spinner_with_message():
            pdf_path = download_pdf(active_id)
        
        # Verify the file actually exists on disk now
        if not os.path.exists(pdf_path):
            st.error("Failed to download the PDF. Please check your internet connection.")
            return
            
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return

    # --- STEP 2: SPLIT SCREEN LAYOUT ---
    col_pdf, col_notepad = st.columns([0.5, 0.5])

    with col_pdf:
        st.subheader("Source Material")
        
        # Controls
        c1, c2 = st.columns([0.4, 0.6])
        with c1:
            page_num = st.number_input("Page", min_value=1, max_value=len(doc), value=1) - 1
        with c2:
            crop_mode = st.toggle("✂️ Snag Mode", help="Draw boxes to instantly clip images to your wall.")

        page = doc[page_num]

        if crop_mode:
            if st_cropper is None:
                st.warning(
                    "Snag Mode requires `streamlit-cropper`. Install it with "
                    "`pip install streamlit-cropper` or `pip install -r requirements.txt`."
                )
                return

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            st.info("Drag to select area. Click button to Snag.")
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            
            if st.button("Add to Wall", width="stretch"):
                buffered = io.BytesIO()
                cropped_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:100%; border-radius:5px; margin-bottom:10px;">'
                
                save_research_note(user_id, img_html, active_id)
                st.toast("Snippet Snagged!")
                st.rerun()
        else:
            # Read the local PDF file we just downloaded
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            
            # Create an HTML embedding string
            # We include the '#page=' parameter to maintain your page navigation
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={page_num+1}" width="100%" height="800" type="application/pdf">'
            
            # Render it using markdown
            st.markdown(pdf_display, unsafe_allow_html=True)

    # --- STEP 3: NOTEPAD ---
    with col_notepad:
        st.subheader("Brainstorm")
    
        # Manual Entry
        with st.form("brainstorm_form", clear_on_submit=True):
            note = st.text_area("Add a thought...", placeholder="Notes, questions, or raw text snippets...", height=100)
            if st.form_submit_button("Append Note", width="stretch"):
                if note:
                    save_research_note(user_id, f"<p>{note}</p>", active_id)
                    st.rerun()

        # Display the wall 
        notes = get_all_notes(user_id)
        
        st.markdown("---")
        with st.container(height=600):
            if not notes:
                st.caption("Wall is empty. Start snagging or typing.")
            for content, aid, ts in notes:
                # We use markdown with unsafe_allow_html to render the base64 images we saved
                st.markdown(content, unsafe_allow_html=True)
                # Replace st.divider() with this for a tight, 5px margin
                st.markdown("<hr style='margin-top:0; margin-bottom: 5px; border: 0.5px solid #eee;'>", unsafe_allow_html=True)

        notes = get_all_notes(user_id)

        st.markdown("---")

        if notes:
            md_content = build_research_export(notes, active_id)

            st.download_button(
                label="Download as Markdown",
                data=md_content,
                file_name=f"research_{active_id}.md",
                mime="text/markdown"
            )

            st.download_button(
                label="Download as TXT",
                data=md_content,
                file_name=f"research_{active_id}.txt",
                mime="text/plain"
            )

def build_research_export(notes, active_id):
    md_lines = []

    md_lines.append(f"# Research Lab Notes")
    md_lines.append(f"Paper: {active_id}")
    md_lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append("\n---\n")

    for content, aid, ts in notes:
        # crude detection: image vs text
        if "<img" in content:
            md_lines.append("### Snagged Image\n")
            md_lines.append(content)
            md_lines.append("\n")
        else:
            # strip simple paragraph wrapper if present
            cleaned = content.replace("<p>", "").replace("</p>", "")
            md_lines.append(cleaned)
            md_lines.append("\n")

        md_lines.append("---\n")

    return "\n".join(md_lines)