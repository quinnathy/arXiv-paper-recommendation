import streamlit as st
import requests
import os
import fitz  # PyMuPDF

def download_pdf(arxiv_id, data_dir="data/temp_pdfs"):
    """
    Downloads the PDF from ArXiv if it doesn't exist locally.
    Returns the full path to the downloaded file.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    pdf_path = os.path.join(data_dir, f"{arxiv_id}.pdf")
    
    # Only download if we don't have it already
    if not os.path.exists(pdf_path):
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download PDF from ArXiv: {e}")
            return None
            
    return pdf_path

def snag_and_drop_router(arxiv_id, page_num, bbox=None):
    """
    The backend for the 'Frantic Snag' mode.
    Currently just a pass-through for the UI to get the PDF path.
    """
    pdf_path = download_pdf(arxiv_id)
    return pdf_path

# Keep this for backward compatibility with your main app routing if needed
def get_paper_markdown(arxiv_id, use_mock=True):
    return f"# Researching {arxiv_id}\nUse Snag Mode to clip images."