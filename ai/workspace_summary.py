"""OpenAI-backed summaries for Workspace papers."""

from __future__ import annotations

import os
from io import BytesIO

import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_SUMMARY_MODEL = "gpt-5.4-mini"
MAX_PAPER_TEXT_CHARS = 45000
PDF_DOWNLOAD_TIMEOUT = 30


def resolve_summary_model(model: str | None = None) -> str:
    return model or os.getenv("OPENAI_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL)


def _download_pdf(arxiv_id: str) -> bytes:
    response = requests.get(
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        timeout=PDF_DOWNLOAD_TIMEOUT,
    )
    response.raise_for_status()
    return response.content


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. Install it with "
            "`pip install pymupdf` or `pip install -r requirements.txt`."
        ) from exc

    with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as document:
        pages = [page.get_text("text") for page in document]
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    return " ".join(text.split())


def _paper_block(index: int, paper: dict) -> str:
    arxiv_id = paper.get("arxiv_id") or paper.get("id") or "unknown"
    title = paper.get("title") or arxiv_id
    abstract = (paper.get("abstract") or "").strip()

    try:
        full_text = _extract_pdf_text(_download_pdf(arxiv_id))
    except Exception:
        full_text = abstract or "[PDF extraction failed and no abstract is available]"

    if len(full_text) > MAX_PAPER_TEXT_CHARS:
        full_text = full_text[:MAX_PAPER_TEXT_CHARS].rstrip() + "..."

    return (
        f"Paper {index}\n"
        f"ID: {arxiv_id}\n"
        f"Title: {title}\n"
        f"Text:\n{full_text}"
    )


def _extract_output_text(payload: dict) -> str:
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                value = content.get("text")
                if isinstance(value, str):
                    chunks.append(value)

    return "\n".join(chunks).strip()


def summarize_workspace(
    papers: list[dict],
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = 60,
) -> str:
    """Summarize and synthesize the papers currently added to the Workspace."""
    if not papers:
        raise ValueError("Add at least one paper to the Workspace before summarizing.")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")

    paper_text = "\n\n---\n\n".join(
        _paper_block(i, paper) for i, paper in enumerate(papers, start=1)
    )
    prompt = (
        "Read the following arXiv papers from the user's Workspace. The text was "
        "extracted from PDFs, so ignore extraction artifacts such as repeated headers, "
        "page numbers, reference lists, and broken line wrapping when they are not "
        "relevant.\n\n"
        "Return a clear, organized Markdown synthesis of the workspace as a whole. "
        "Do not simply summarize each paper one after another. Treat the papers as "
        "a small research collection, usually centered on the same broad topic or "
        "field but covering different approaches, methods, assumptions, or problem "
        "settings. Your goal is to help the user understand the ideas present across "
        "the workspace, connect the papers constructively, and suggest ways those "
        "ideas could lead to a new or enhanced research direction. If a paper is an "
        "outlier and not closely connected to the others, mention it."
        "If the papers just don't fit together cohesively at all, "
        "don't try to force connections in the rest of the query, instead summarize"
        "the papers individually and don't add the Shared Ideas / Connections section.\n\n"
        "When referring to a paper, use its actual title or a short readable version "
        "of its title. Do not refer to papers primarily by arXiv IDs such as "
        "`2604.16209`. Use arXiv IDs only in parentheses when disambiguation is "
        "necessary.\n\n"
        "Use this structure:\n"
        "## Thesis\n"
        "Give the main overview of the workspace in 2-4 sentences: the field or "
        "subfield it covers, the central problem family, and the broad direction the "
        "papers point toward together. Use concrete problem or idea names that can "
        "later become labels for visualization edges between papers.\n\n"
        "## Paper Roles\n"
        "Briefly identify how each paper fits into the field(s) the workspace covers. "
        "For each paper, use its title or a short title, then summarize its role and "
        "main research method, such as a modeling approach, benchmark, dataset, "
        "algorithm, theory, application study, evaluation, or survey.\n\n"
        "## Shared Research Problems / Ideas\n"
        "Synthesize the research problems, thesis-level ideas, and motivations that "
        "recur across the workspace. Name the specific papers connected by each "
        "shared problem or idea, and prefer concise reusable labels over generic "
        "claims.\n\n"
        "## Similarities\n"
        "Identify concrete similarities between papers. Group them by assumptions, "
        "methodology, datasets, and evaluations when those details are present. Name "
        "the specific papers connected by each similarity so a downstream graph "
        "agent can turn them into edges.\n\n"
        "## Tensions / Gaps\n"
        "Identify open problems, limitations, tradeoffs, conflicting assumptions, "
        "missing experiments, or places where one paper's approach leaves room for "
        "another paper's ideas.\n\n"
        "## Possible New Directions\n"
        "Propose 1-3 plausible research directions that build off the papers as a "
        "group. For each direction, include the core idea, how it combines or extends "
        "the papers, why it could be useful, and a concrete first experiment or "
        "prototype the user could try. Keep the ideas grounded in the provided text.\n\n"
        "## What To Pay Attention To\n"
        "List practical reading notes and questions the user should keep in mind "
        "while moving between the papers.\n\n"
        "Use only the provided text. Do not invent results, citations, or claims that "
        "are not supported by the extracted paper text. The research directions should "
        "be hypotheses or project ideas inspired by the workspace, not claims that "
        "the papers already proved them.\n\n"
        "Do not add any closing offers, follow-up suggestions, or extra lines like "
        "'if you want, I can make a table.' Just answer using the requested sections.\n\n"
        f"{paper_text}"
    )

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": resolve_summary_model(model),
            "instructions": (
                "You are a careful research assistant helping a student understand "
                "papers in a reading workspace."
            ),
            "input": prompt,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    summary = _extract_output_text(response.json())
    if not summary:
        raise RuntimeError("The summary response did not include any text.")

    return summary
