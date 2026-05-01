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


def test_visualization_view_loads_map_for_current_workspace(monkeypatch):
    fake_st = _FakeStreamlit()
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
