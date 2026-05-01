"""OpenAI-backed paper connection extraction for Workspace visualization."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from ai.workspace_summary import OPENAI_RESPONSES_URL, _extract_output_text


DEFAULT_CONNECTION_MODEL = "gpt-5.4-mini"
MAX_CONNECTIONS = 12
VALID_CONNECTION_TYPES = {
    "thesis",
    "assumption",
    "methodology",
    "dataset",
    "evaluation",
}
VALID_SUMMARY_SECTIONS = {
    "Thesis",
    "Shared Research Problems / Ideas",
    "Similarities",
}


def resolve_connection_model(model: str | None = None) -> str:
    return model or os.getenv("OPENAI_CONNECTION_MODEL", DEFAULT_CONNECTION_MODEL)


def _paper_context(papers: list[dict]) -> str:
    blocks = []
    for paper in papers:
        arxiv_id = paper.get("arxiv_id") or paper.get("id") or "unknown"
        title = paper.get("title") or arxiv_id
        abstract = " ".join(str(paper.get("abstract") or "").split())
        blocks.append(
            f"ID: {arxiv_id}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract or '[No abstract available]'}"
        )
    return "\n\n---\n\n".join(blocks)


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Connection response must be a JSON object.")
    return payload


def parse_workspace_connections(
    payload: dict[str, Any],
    papers: list[dict],
    max_connections: int = MAX_CONNECTIONS,
) -> list[dict[str, Any]]:
    paper_ids = {
        str(paper.get("arxiv_id") or paper.get("id") or "") for paper in papers
    }
    parsed: list[dict[str, Any]] = []
    for item in payload.get("connections", []):
        if not isinstance(item, dict):
            continue

        source = str(item.get("source") or "")
        target = str(item.get("target") or "")
        connection_type = str(item.get("type") or "")
        if (
            not source
            or not target
            or source == target
            or source not in paper_ids
            or target not in paper_ids
            or connection_type not in VALID_CONNECTION_TYPES
        ):
            continue

        section = str(item.get("summary_section") or "Similarities")
        if section not in VALID_SUMMARY_SECTIONS:
            section = "Similarities"

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        parsed.append(
            {
                "source": source,
                "target": target,
                "type": connection_type,
                "name": str(item.get("name") or connection_type.title()),
                "summary_section": section,
                "rationale": str(item.get("rationale") or ""),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )

    parsed.sort(key=lambda edge: edge["confidence"], reverse=True)
    return parsed[:max_connections]


def generate_workspace_connections(
    papers: list[dict],
    summary_text: str | None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = 60,
    abstract_only: bool = False,
) -> list[dict[str, Any]]:
    if len(papers) < 2:
        return []

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")

    summary_context = (
        "[No cached summary was available. Use only the paper abstracts below.]"
        if abstract_only or not summary_text
        else summary_text
    )
    prompt = (
        "You are building edges for a research-paper visualization. Read the "
        "workspace summary and paper abstracts, then identify only the strongest "
        "paper-to-paper connections. Return strict JSON only, with no Markdown.\n\n"
        "Use these connection types exactly: thesis, assumption, methodology, "
        "dataset, evaluation. Map thesis/problem/idea links to the Thesis or "
        "Shared Research Problems / Ideas sections. Map assumptions, methods, "
        "datasets, and evaluation links to Similarities when appropriate.\n\n"
        "Rules:\n"
        "- Return at most 12 connections.\n"
        "- Use only the provided paper IDs in source and target.\n"
        "- Prefer fewer, stronger edges over dense graph coverage.\n"
        "- For repeated paper pairs, include multiple edge types only when the "
        "connection names are meaningfully different.\n"
        "- The name must be a short hover label, ideally 2-6 words.\n"
        "- The rationale must be one sentence.\n\n"
        "JSON shape:\n"
        '{"connections":[{"source":"arxiv_id","target":"arxiv_id",'
        '"type":"thesis|assumption|methodology|dataset|evaluation",'
        '"name":"short hover label","summary_section":"Thesis|Shared Research '
        'Problems / Ideas|Similarities","rationale":"one sentence",'
        '"confidence":0.0}]}\n\n'
        f"Workspace summary:\n{summary_context}\n\n"
        f"Workspace papers:\n{_paper_context(papers)}"
    )

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": resolve_connection_model(model),
            "instructions": (
                "You extract concise, evidence-grounded graph edges between "
                "papers for an interactive research visualization."
            ),
            "input": prompt,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    text = _extract_output_text(response.json())
    if not text:
        raise RuntimeError("The connection response did not include any text.")
    return parse_workspace_connections(_loads_json_object(text), papers)

