import streamlit as st
from streamlit_cropper import st_cropper
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import os

# Ensure this import matches the filename and function exactly
from pipeline.transcribe import download_pdf, snag_and_drop_router
from user.db import get_all_notes, save_research_note

def render_research_mode(index=None):
    user_id = st.session_state["user_id"]
    active_id = st.session_state.get("active_arxiv_id", "No Paper Selected")

    st.title("Research Lab")
    
    if active_id == "No Paper Selected":
        st.info("Select a paper from your feed to start a session.")
        return

    # --- STEP 1: DOWNLOAD CHECK ---
    # We do this before setting up columns to avoid the FileNotFoundError
    try:
        with st.spinner(f"Fetching paper {active_id} from ArXiv..."):
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
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            st.info("Drag to select area. Click button to Snag.")
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            
            if st.button("✨ SNAG TO WALL", use_container_width=True):
                buffered = io.BytesIO()
                cropped_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:100%; border-radius:5px; margin-bottom:10px;">'
                
                save_research_note(user_id, img_html, active_id)
                st.toast("Snippet Snagged!")
                st.rerun()
        else:
            # Note: local file path with iframe is tricky in Streamlit, 
            # so we keep using the ArXiv URL for the scrollable preview.
            pdf_url = f"https://arxiv.org/pdf/{active_id}.pdf#page={page_num+1}"
            st.components.v1.iframe(pdf_url, height=800)

    # --- STEP 3: NOTEPAD ---
    with col_notepad:
        st.subheader("📓 Infinite Wall")
        # ... (rest of your notepad/wall display code remains the same)
        
        # Frantic Manual Entry
        with st.form("brainstorm_form", clear_on_submit=True):
            note = st.text_area("Type a thought...", placeholder="Notes, questions, or raw text snippets...", height=100)
            if st.form_submit_button("Append Note", use_container_width=True):
                if note:
                    save_research_note(user_id, f"<p>{note}</p>", active_id)
                    st.rerun()

        # Display the wall (Frantic vibe: Everything appears in a long scrollable column)
        notes = get_all_notes(user_id)
        
        st.markdown("---")
        # Scrollable container for the 'frantic' feeling
        with st.container(height=600):
            if not notes:
                st.caption("Wall is empty. Start snagging or typing.")
            for content, aid, ts in notes:
                # We use markdown with unsafe_allow_html to render the base64 images we saved
                st.markdown(content, unsafe_allow_html=True)
                st.caption(f"📎 {ts[11:16]}") # Show just the time for the 'frantic' vibe
                st.divider()