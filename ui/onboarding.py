"""Onboarding page for new users.

Shown when the user has not yet completed the initial setup.
Collects a display name and topic preferences, then creates a user profile
with an initial embedding derived from the selected category centroids.
"""

from __future__ import annotations

from pipeline.index import PaperIndex


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    """Render the onboarding page for a new (not yet onboarded) user.

    Page layout:
        1. App title + one-line description.
        2. Name input field (st.text_input).
        3. topic_selector() widget showing available arXiv categories.
        4. "Start reading" button.

    On button click:
        - Validate: name must be non-empty, at least 1 topic must be selected.
        - Compute initial embedding:
            embedding = init_embedding_from_topics(selected, index.category_centroids)
        - Create user in database:
            user_id = create_user(name, embedding)
        - Update st.session_state:
            user_id, user_embedding, onboarded = True
        - Call st.rerun() to transition to the daily feed.

    Args:
        index: The loaded PaperIndex (needed for category_centroids).
        db_path: Path to the SQLite database.
    """
    raise NotImplementedError
