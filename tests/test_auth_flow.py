"""Comprehensive tests for signup/login/guest auth behaviors.

These tests focus on backend correctness for returning-user handling:
- Account creation with username/password
- Credential authentication and edge cases
- Session login/logout lifecycle
- Compatibility with guest users (no credentials)
"""

from __future__ import annotations

import numpy as np
import pytest

from user.db import authenticate_user, create_user, get_user, init_db
from user import session as user_session


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _centroids(k_u: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(k_u):
        rows.append(_unit(rng.standard_normal(768).astype(np.float32)))
    return np.stack(rows)


@pytest.fixture
def db_path(tmp_path) -> str:
    db = str(tmp_path / "auth.db")
    init_db(db)
    return db


@pytest.fixture
def fake_streamlit(monkeypatch):
    class DummyStreamlit:
        def __init__(self):
            self.session_state = {}

    dummy = DummyStreamlit()
    monkeypatch.setattr(user_session, "st", dummy)
    return dummy


class TestAccountCreationAndAuthentication:
    def test_signup_and_authenticate_success(self, db_path):
        uid = create_user(
            "alice",
            _centroids(2, seed=1),
            k_u=2,
            diversity=0.6,
            username="alice",
            password="alice-password-123",
        )

        user = get_user(uid)
        assert user is not None
        assert user["username"] == "alice"

        authed = authenticate_user("alice", "alice-password-123")
        assert authed is not None
        assert authed["user_id"] == uid
        assert authed["username"] == "alice"

    def test_duplicate_username_rejected_case_insensitive(self, db_path):
        create_user(
            "Alice",
            _centroids(1, seed=2),
            k_u=1,
            username="Alice",
            password="password-123",
        )
        with pytest.raises(ValueError, match="Username already exists"):
            create_user(
                "alice second",
                _centroids(1, seed=3),
                k_u=1,
                username="alice",
                password="password-xyz",
            )

    def test_wrong_password_and_unknown_user_fail(self, db_path):
        create_user(
            "bob",
            _centroids(1, seed=4),
            k_u=1,
            username="bob",
            password="right-password",
        )
        assert authenticate_user("bob", "wrong-password") is None
        assert authenticate_user("does-not-exist", "any") is None

    def test_username_lookup_is_case_insensitive(self, db_path):
        uid = create_user(
            "charlie",
            _centroids(1, seed=5),
            k_u=1,
            username="charlie",
            password="charlie-password",
        )
        authed = authenticate_user("ChArLiE", "charlie-password")
        assert authed is not None
        assert authed["user_id"] == uid

    def test_guest_user_has_no_credentials(self, db_path):
        uid = create_user(
            "Guest One",
            _centroids(1, seed=6),
            k_u=1,
            diversity=0.5,
            username=None,
            password=None,
        )
        user = get_user(uid)
        assert user is not None
        assert user["username"] is None
        assert authenticate_user("Guest One", "anything") is None


class TestSessionLoginLifecycle:
    def test_login_with_credentials_populates_session(self, db_path, fake_streamlit):
        uid = create_user(
            "dana",
            _centroids(2, seed=7),
            k_u=2,
            diversity=0.7,
            username="dana",
            password="dana-password",
        )
        assert user_session.login_with_credentials("dana", "dana-password")

        ss = fake_streamlit.session_state
        assert ss["user_id"] == uid
        assert ss["onboarded"] is True
        assert ss["user_k_u"] == 2
        assert ss["user_centroids"].shape == (2, 768)
        assert np.isclose(ss["user_diversity"], 0.7)

    def test_login_with_credentials_resets_feed_cache(self, db_path, fake_streamlit):
        create_user(
            "erin",
            _centroids(1, seed=8),
            k_u=1,
            username="erin",
            password="erin-password",
        )
        fake_streamlit.session_state["todays_recs"] = [{"id": "x"}]
        fake_streamlit.session_state["responded"] = {"x"}
        fake_streamlit.session_state["learning_workspace"] = ["2401.00001"]
        fake_streamlit.session_state["workspace_result_view"] = "visualization"
        fake_streamlit.session_state["query_search_results"] = [{"id": "y"}]

        assert user_session.login_with_credentials("erin", "erin-password")
        assert "todays_recs" not in fake_streamlit.session_state
        assert "responded" not in fake_streamlit.session_state
        assert "learning_workspace" not in fake_streamlit.session_state
        assert "workspace_result_view" not in fake_streamlit.session_state
        assert "query_search_results" not in fake_streamlit.session_state

    def test_login_with_credentials_failure_does_not_onboard(self, db_path, fake_streamlit):
        create_user(
            "frank",
            _centroids(1, seed=9),
            k_u=1,
            username="frank",
            password="frank-password",
        )
        ok = user_session.login_with_credentials("frank", "bad-password")
        assert ok is False
        assert fake_streamlit.session_state.get("onboarded") is not True

    def test_legacy_login_by_user_id_still_works(self, db_path, fake_streamlit):
        uid = create_user(
            "legacy",
            _centroids(1, seed=10),
            k_u=1,
            username="legacy",
            password="legacy-password",
        )
        assert user_session.login_user(uid, db_path)
        assert fake_streamlit.session_state["user_id"] == uid
        assert fake_streamlit.session_state["onboarded"] is True

    def test_logout_clears_auth_state(self, db_path, fake_streamlit):
        create_user(
            "gina",
            _centroids(1, seed=11),
            k_u=1,
            username="gina",
            password="gina-password",
        )
        assert user_session.login_with_credentials("gina", "gina-password")
        fake_streamlit.session_state["todays_recs"] = [{"id": "z"}]
        fake_streamlit.session_state["responded"] = {"z"}
        fake_streamlit.session_state["learning_workspace"] = ["2401.00002"]
        fake_streamlit.session_state["workspace_similar_papers"] = [{"id": "a"}]
        fake_streamlit.session_state["workspace_result_view"] = "similar"
        fake_streamlit.session_state["workspace_map_paper_threshold"] = 0.8
        fake_streamlit.session_state["query_search_results"] = [{"id": "b"}]
        fake_streamlit.session_state["active_arxiv_id"] = "2401.00002"
        fake_streamlit.session_state["active_tab_value"] = "Research Lab"

        user_session.logout_user()

        ss = fake_streamlit.session_state
        assert ss["user_id"] is None
        assert ss["user_centroids"] is None
        assert ss["user_k_u"] is None
        assert ss["user_diversity"] is None
        assert ss["onboarded"] is False
        assert "todays_recs" not in ss
        assert "responded" not in ss
        assert "learning_workspace" not in ss
        assert "workspace_similar_papers" not in ss
        assert "workspace_result_view" not in ss
        assert "workspace_map_paper_threshold" not in ss
        assert "query_search_results" not in ss
        assert "active_arxiv_id" not in ss
        assert "active_tab_value" not in ss
