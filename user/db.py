"""SQLite database schema and CRUD operations for user profiles and feedback.

Two tables:
    users    — stores user profile with serialized embedding blob.
    feedback — logs every like/save/skip interaction for analytics and seen_ids.

No ORM; uses plain sqlite3. All writes are synchronous (Streamlit is
single-user per session, so no threading is needed).

Embedding serialization:
    Store:  embedding.astype(np.float32).tobytes()
    Load:   np.frombuffer(blob, dtype=np.float32)
"""

from __future__ import annotations

import numpy as np


def init_db(db_path: str = "data/arxiv_rec.db") -> None:
    """Create the users and feedback tables if they do not exist.

    Schema — users:
        user_id      TEXT PRIMARY KEY  — UUID string
        display_name TEXT              — user-chosen display name
        embedding    BLOB              — numpy float32 array as raw bytes
        created_at   TEXT              — ISO datetime
        last_active  TEXT              — ISO datetime

    Schema — feedback:
        id           INTEGER PRIMARY KEY AUTOINCREMENT
        user_id      TEXT               — FK to users.user_id
        arxiv_id     TEXT               — arXiv paper ID
        signal       TEXT               — "like" | "save" | "skip"
        cluster_id   INTEGER            — cluster of the paper (for analytics)
        score        REAL               — recommendation score at time of serving
        created_at   TEXT               — ISO datetime

    Args:
        db_path: Path to the SQLite database file.

    Implementation:
        - Connect to db_path with sqlite3.connect().
        - Execute CREATE TABLE IF NOT EXISTS for both tables.
        - Commit and close.
    """
    raise NotImplementedError


def create_user(display_name: str, embedding: np.ndarray) -> str:
    """Create a new user record in the database.

    Args:
        display_name: The user's display name.
        embedding: The user's initial embedding, shape (768,), float32, unit-norm.

    Returns:
        The generated user_id (UUID string).

    Implementation:
        - Generate a UUID4 string.
        - Serialize embedding as embedding.astype(np.float32).tobytes().
        - INSERT into users with current ISO datetime for created_at and last_active.
        - Return the user_id.
    """
    raise NotImplementedError


def get_user(user_id: str) -> dict | None:
    """Retrieve a user record by user_id.

    Args:
        user_id: The UUID string of the user.

    Returns:
        Dict with keys: user_id, display_name, embedding (np.ndarray float32),
        created_at, last_active.
        Returns None if user not found.

    Implementation:
        - SELECT * FROM users WHERE user_id=?
        - Decode embedding BLOB → np.frombuffer(blob, dtype=np.float32).
        - Return as dict.
    """
    raise NotImplementedError


def update_embedding(user_id: str, embedding: np.ndarray) -> None:
    """Update a user's embedding in the database.

    Args:
        user_id: The UUID string of the user.
        embedding: The new embedding, shape (768,), float32, unit-norm.

    Implementation:
        - Serialize embedding as embedding.astype(np.float32).tobytes().
        - UPDATE users SET embedding=?, last_active=? WHERE user_id=?
    """
    raise NotImplementedError


def log_feedback(
    user_id: str,
    arxiv_id: str,
    signal: str,
    cluster_id: int,
    score: float,
) -> None:
    """Log a user feedback event (like/save/skip) to the feedback table.

    Args:
        user_id: The UUID string of the user.
        arxiv_id: The arXiv paper ID.
        signal: One of "like", "save", "skip".
        cluster_id: The cluster ID of the paper.
        score: The recommendation score at the time of serving.

    Implementation:
        - INSERT INTO feedback (user_id, arxiv_id, signal, cluster_id, score, created_at)
        - Use current ISO datetime for created_at.
    """
    raise NotImplementedError


def get_seen_ids(user_id: str) -> set[str]:
    """Get the set of all arXiv paper IDs this user has interacted with.

    Args:
        user_id: The UUID string of the user.

    Returns:
        Set of arxiv_id strings from all feedback rows for this user,
        regardless of signal type. Once seen, a paper is never re-served.

    Implementation:
        - SELECT arxiv_id FROM feedback WHERE user_id=?
        - Return as a set.
    """
    raise NotImplementedError
