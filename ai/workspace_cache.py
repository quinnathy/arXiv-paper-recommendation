"""Disk cache helpers for Workspace AI outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_CACHE_ROOT = Path("data/workspace_cache")


def workspace_cache_signature(papers: list[dict]) -> str:
    ids = [str(paper.get("arxiv_id") or paper.get("id") or "") for paper in papers]
    payload = json.dumps(ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def workspace_cache_dir(
    papers: list[dict],
    cache_root: str | Path = WORKSPACE_CACHE_ROOT,
) -> Path:
    return Path(cache_root) / workspace_cache_signature(papers)


def workspace_summary_cache_paths(
    papers: list[dict],
    cache_root: str | Path = WORKSPACE_CACHE_ROOT,
) -> dict[str, Path]:
    directory = workspace_cache_dir(papers, cache_root)
    return {
        "dir": directory,
        "summary": directory / "summary.md",
        "metadata": directory / "metadata.json",
        "connections": directory / "connections.json",
    }


def write_workspace_summary_cache(
    papers: list[dict],
    summary: str,
    model: str,
    cache_root: str | Path = WORKSPACE_CACHE_ROOT,
) -> dict[str, Path]:
    paths = workspace_summary_cache_paths(papers, cache_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(summary, encoding="utf-8")

    metadata = {
        "workspace_signature": workspace_cache_signature(papers),
        "workspace_ids": [
            str(paper.get("arxiv_id") or paper.get("id") or "") for paper in papers
        ],
        "paper_titles": [
            str(paper.get("title") or paper.get("arxiv_id") or paper.get("id") or "")
            for paper in papers
        ],
        "summary_model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def read_workspace_summary_cache(
    papers: list[dict],
    cache_root: str | Path = WORKSPACE_CACHE_ROOT,
) -> str | None:
    path = workspace_summary_cache_paths(papers, cache_root)["summary"]
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def write_workspace_connections_cache(
    papers: list[dict],
    connections: list[dict[str, Any]],
    cache_root: str | Path = WORKSPACE_CACHE_ROOT,
) -> Path:
    paths = workspace_summary_cache_paths(papers, cache_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "workspace_signature": workspace_cache_signature(papers),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "connections": connections,
    }
    paths["connections"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths["connections"]

