from contextlib import nullcontext
from types import SimpleNamespace

import ui.learning_mode as learning_mode


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self, thresholds=None):
        self.session_state = {
            "workspace_result_view": "visualization",
        }
        if thresholds:
            self.session_state.update(thresholds)
        self.info_messages = []

    def subheader(self, *_args, **_kwargs):
        pass

    def divider(self):
        pass

    def expander(self, *_args, **_kwargs):
        return nullcontext()

    def columns(self, count, **_kwargs):
        return [_FakeColumn() for _ in range(count)]

    def slider(self, _label, *, value, key, **_kwargs):
        self.session_state[key] = value
        return value

    def info(self, message):
        self.info_messages.append(message)


def test_visualization_view_reloads_map_from_current_slider_values(monkeypatch):
    fake_st = _FakeStreamlit(
        {
            "workspace_map_paper_threshold": 0.6,
            "workspace_map_concept_threshold": 0.25,
        }
    )
    calls = []

    def fake_load(index, workspace_papers, **kwargs):
        calls.append((index, workspace_papers, kwargs))

    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "_load_workspace_concept_map", fake_load)

    index = SimpleNamespace(concept_embeddings={"concept": object()})
    workspace_papers = [{"arxiv_id": "1234.5678"}]

    learning_mode._render_workspace_result_panel(index, workspace_papers, set())

    assert len(calls) == 1
    assert calls[0][0] is index
    assert calls[0][1] == workspace_papers
    assert calls[0][2] == {
        "paper_similarity_threshold": 0.6,
        "concept_similarity_threshold": 0.25,
    }


def test_visualization_view_defaults_link_thresholds_to_point_five(monkeypatch):
    fake_st = _FakeStreamlit()
    calls = []

    def fake_load(index, workspace_papers, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(learning_mode, "st", fake_st)
    monkeypatch.setattr(learning_mode, "_load_workspace_concept_map", fake_load)

    index = SimpleNamespace(concept_embeddings={"concept": object()})
    workspace_papers = [{"arxiv_id": "1234.5678"}]

    learning_mode._render_workspace_result_panel(index, workspace_papers, set())

    assert calls == [
        {
            "paper_similarity_threshold": 0.5,
            "concept_similarity_threshold": 0.5,
        }
    ]
