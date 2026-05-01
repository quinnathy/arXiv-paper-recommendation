from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import ui.learning_mode as learning_mode


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self):
        self.session_state = {
            "workspace_result_view": "visualization",
        }
        self.info_messages = []
        self.warning_messages = []
        self.caption_messages = []
        self.buttons = []
        self.secrets = {}

    def subheader(self, *_args, **_kwargs):
        pass

    def divider(self):
        pass

    def expander(self, *_args, **_kwargs):
        return nullcontext()

    def columns(self, count, **_kwargs):
        return [_FakeColumn() for _ in range(count)]

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)

    def caption(self, message):
        self.caption_messages.append(message)

    def metric(self, *_args, **_kwargs):
        pass

    def plotly_chart(self, *_args, **_kwargs):
        pass

    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return False

    def markdown(self, *_args, **_kwargs):
        pass


def test_visualization_view_loads_map_for_current_workspace(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "workspace_summary": "summary",
            "workspace_summary_signature": ("1234.5678",),
        }
    )
    calls = []

    def fake_load(index, workspace_papers):
        calls.append((index, workspace_papers))

    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "_load_workspace_concept_map", fake_load)

    index = SimpleNamespace(concept_embeddings={"concept": object()})
    workspace_papers = [{"arxiv_id": "1234.5678"}]

    learning_mode._render_workspace_result_panel(index, workspace_papers, set())

    assert len(calls) == 1
    assert calls[0][0] is index
    assert calls[0][1] == workspace_papers


def test_visualization_view_requires_current_summary(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(learning_mode, "st", fake_st)

    learning_mode._render_workspace_result_panel(
        SimpleNamespace(),
        [{"arxiv_id": "1234.5678"}],
        set(),
    )

    assert fake_st.info_messages == ["Run Summarize before opening the visualization."]


def test_workspace_summary_ready_requires_current_signature(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(learning_mode, "st", fake_st)

    papers = [{"arxiv_id": "workspace-1"}]
    assert learning_mode._workspace_summary_ready(papers) is False

    fake_st.session_state.update(
        {
            "workspace_summary": "summary",
            "workspace_summary_signature": ("other-workspace",),
        }
    )
    assert learning_mode._workspace_summary_ready(papers) is False

    fake_st.session_state["workspace_summary_signature"] = ("workspace-1",)
    assert learning_mode._workspace_summary_ready(papers) is True


def test_load_workspace_summary_writes_disk_cache(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "loading_spinner_with_message", nullcontext)
    monkeypatch.setattr(
        learning_mode,
        "summarize_workspace",
        lambda papers, api_key=None: "## Thesis\nCached summary",
    )
    captured = {}

    def fake_write(papers, summary, model):
        captured["papers"] = papers
        captured["summary"] = summary
        captured["model"] = model

    monkeypatch.setattr(learning_mode, "write_workspace_summary_cache", fake_write)

    papers = [{"arxiv_id": "workspace-1", "title": "Workspace Paper"}]
    learning_mode._load_workspace_summary(papers)

    assert fake_st.session_state["workspace_summary"] == "## Thesis\nCached summary"
    assert fake_st.session_state["workspace_summary_signature"] == ("workspace-1",)
    assert captured["papers"] == papers
    assert captured["summary"] == "## Thesis\nCached summary"


def test_workspace_visualization_falls_back_to_abstracts_when_cache_missing(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "workspace_summary": "summary in session",
            "workspace_summary_signature": ("workspace-1", "workspace-2"),
        }
    )
    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "loading_spinner_with_message", nullcontext)
    monkeypatch.setattr(learning_mode, "read_workspace_summary_cache", lambda papers: None)

    captured = {}

    def fake_generate(papers, summary_text, api_key=None, abstract_only=False):
        captured["summary_text"] = summary_text
        captured["abstract_only"] = abstract_only
        return [
            {
                "source": "workspace-1",
                "target": "workspace-2",
                "type": "thesis",
                "name": "Shared problem",
            }
        ]

    monkeypatch.setattr(learning_mode, "generate_workspace_connections", fake_generate)
    monkeypatch.setattr(learning_mode, "write_workspace_connections_cache", lambda *args: None)

    connections, source = learning_mode._load_workspace_connections(
        [
            {"arxiv_id": "workspace-1", "title": "First"},
            {"arxiv_id": "workspace-2", "title": "Second"},
        ]
    )

    assert source == "abstracts"
    assert captured == {"summary_text": None, "abstract_only": True}
    assert connections[0]["name"] == "Shared problem"


def test_workspace_see_more_uses_onboarding_profile_and_exclusions(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "user_id": "user-1",
            "user_diversity": 0.7,
            "shown_ids": {"already-shown"},
        }
    )
    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "get_seen_ids", lambda user_id: {"skipped"})

    captured = {}
    profile_centroids = np.array([[0.0, 1.0]], dtype=np.float32)

    def fake_init(seeds):
        captured["seed_labels"] = [seed.label for seed in seeds]
        return SimpleNamespace(centroids=profile_centroids)

    def fake_recommend(user_centroids, seen_ids, index, diversity, n):
        captured["user_centroids"] = user_centroids
        captured["seen_ids"] = seen_ids
        captured["diversity"] = diversity
        captured["n"] = n
        return [{"id": "fresh-paper", "title": "Fresh Paper"}]

    monkeypatch.setattr(learning_mode, "init_user_profile_v2", fake_init)
    monkeypatch.setattr(learning_mode, "recommend", fake_recommend)

    index = SimpleNamespace(
        paper_meta=[
            {"id": "workspace-1"},
            {"id": "workspace-2"},
            {"id": "fresh-paper"},
        ],
        embeddings=np.array(
            [
                [3.0, 4.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    workspace_papers = [
        {"arxiv_id": "workspace-1", "title": "First workspace paper"},
        {"arxiv_id": "workspace-2", "title": "Second workspace paper"},
    ]

    recs = learning_mode._find_workspace_similar_papers(index, workspace_papers)

    assert recs == [{"id": "fresh-paper", "title": "Fresh Paper"}]
    assert captured["seed_labels"] == [
        "First workspace paper",
        "Second workspace paper",
    ]
    assert captured["user_centroids"] is profile_centroids
    assert captured["seen_ids"] == {
        "already-shown",
        "skipped",
        "workspace-1",
        "workspace-2",
    }
    assert captured["diversity"] == 0.7
    assert captured["n"] == learning_mode.WORKSPACE_SIMILAR_LIMIT
    assert fake_st.session_state["shown_ids"] == {"already-shown", "fresh-paper"}


def test_workspace_see_more_refreshes_for_same_workspace(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "workspace_similar_papers": [{"id": "old-paper"}],
            "workspace_similar_signature": ("workspace-1",),
        }
    )
    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "loading_spinner_with_message", nullcontext)
    monkeypatch.setattr(
        learning_mode,
        "_find_workspace_similar_papers",
        lambda index, workspace_papers: [{"id": "new-paper"}],
    )

    learning_mode._load_workspace_similar_papers(
        SimpleNamespace(),
        [{"arxiv_id": "workspace-1"}],
    )

    assert fake_st.session_state["workspace_similar_papers"] == [{"id": "new-paper"}]
    assert fake_st.session_state["workspace_similar_signature"] == ("workspace-1",)
    assert fake_st.session_state["workspace_similar_requested"] is True
