import json
import shutil
from pathlib import Path

from ai.workspace_cache import (
    read_workspace_summary_cache,
    workspace_cache_signature,
    workspace_summary_cache_paths,
    write_workspace_summary_cache,
)
from ai.workspace_connections import parse_workspace_connections


def _papers():
    return [
        {"arxiv_id": "2401.00001", "title": "First Paper", "abstract": "A"},
        {"arxiv_id": "2401.00002", "title": "Second Paper", "abstract": "B"},
    ]


def test_summary_cache_writes_markdown_and_metadata():
    papers = _papers()
    cache_root = Path("data/workspace_cache_test")
    shutil.rmtree(cache_root, ignore_errors=True)
    paths = write_workspace_summary_cache(
        papers,
        "## Thesis\nA useful synthesis.",
        model="test-model",
        cache_root=cache_root,
    )

    assert paths["summary"].read_text(encoding="utf-8").startswith("## Thesis")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["workspace_signature"] == workspace_cache_signature(papers)
    assert metadata["workspace_ids"] == ["2401.00001", "2401.00002"]
    assert metadata["paper_titles"] == ["First Paper", "Second Paper"]
    assert metadata["summary_model"] == "test-model"
    assert read_workspace_summary_cache(papers, cache_root) == "## Thesis\nA useful synthesis."
    assert workspace_summary_cache_paths(papers, cache_root)["summary"] == paths["summary"]
    shutil.rmtree(cache_root, ignore_errors=True)


def test_parse_workspace_connections_filters_and_limits():
    papers = _papers()
    payload = {
        "connections": [
            {
                "source": "2401.00001",
                "target": "2401.00002",
                "type": "methodology",
                "name": "Shared model",
                "summary_section": "Similarities",
                "rationale": "Both papers use related models.",
                "description": "Both papers use related models for the same task.",
                "confidence": 0.8,
            },
            {
                "source": "2401.00001",
                "target": "missing",
                "type": "dataset",
                "name": "Bad paper id",
            },
            {
                "source": "2401.00001",
                "target": "2401.00002",
                "type": "not-a-type",
                "name": "Bad type",
            },
        ]
    }

    parsed = parse_workspace_connections(payload, papers, max_connections=1)

    assert parsed == [
        {
            "source": "2401.00001",
            "target": "2401.00002",
            "type": "methodology",
            "name": "Shared model",
            "summary_section": "Similarities",
            "rationale": "Both papers use related models.",
            "description": "Both papers use related models for the same task.",
            "confidence": 0.8,
        }
    ]
