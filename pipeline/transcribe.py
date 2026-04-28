import streamlit as st
import requests
import os

# Try to import marker, but provide a fallback for the "Fast Preview"
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    HAS_MARKER = True
except ImportError:
    HAS_MARKER = False

@st.cache_resource
def get_marker_models():
    """Loads the layout and OCR models (takes time, hence cached)."""
    if HAS_MARKER:
        return load_all_models()
    return None

def download_pdf(arxiv_id, data_dir="data/temp_pdfs"):
    """Downloads the PDF from ArXiv for Marker to process."""
    os.makedirs(data_dir, exist_ok=True)
    pdf_path = os.path.join(data_dir, f"{arxiv_id}.pdf")
    
    if not os.path.exists(pdf_path):
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url)
        with open(pdf_path, "wb") as f:
            f.write(response.content)
    return pdf_path

def get_paper_markdown(arxiv_id, use_mock=True):
    """
    The main entry point for Research Mode.
    If use_mock=True, it returns dummy text for UI testing.
    """
    if use_mock:
        return f"""
            # Mock Transcription for {arxiv_id}
                    
            ## Key Insights
            This is a **preview mode** transcription. 
            In production, Marker would analyze the columns and equations here.

            ## Sample Equation
            $L = \\sum_{{i=1}}^{{n}} (y_i - \\hat{{y}}_i)^2$

            ## Notes
            You can copy this text and paste it into your **Infinite Wall** on the right!
                    """

    # Real Marker Logic
    pdf_path = download_pdf(arxiv_id)
    models = get_marker_models()
    
    # Marker processing
    full_text, images, out_metadata = convert_single_pdf(pdf_path, models)
    return full_text