"""SQLite database schema and CRUD operations for user profiles and feedback.

Core tables:
    users    — stores user profile with serialized multi-vector centroids blob.
    feedback — logs every like/save/skip interaction for analytics.
    seen_papers — logs papers served in the daily feed for seen_ids.

No ORM; uses plain sqlite3. All writes are synchronous (Streamlit is
single-user per session, so no threading is needed).

Centroids serialization:
    Store:  centroids.astype(np.float32).tobytes()   # shape (k_u, 768) flattened
    Load:   np.frombuffer(blob, dtype=np.float32).copy().reshape(k_u, 768)
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np

DB_PATH = "data/arxiv_rec.db"


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    return sqlite3.connect(db_path or DB_PATH)


def init_db(db_path: str = DB_PATH) -> None:
    """Create the users and feedback tables if they do not exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    global DB_PATH
    DB_PATH = db_path

    conn = _connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id        TEXT PRIMARY KEY,
            display_name   TEXT NOT NULL,
            username       TEXT,
            password_hash  TEXT,
            centroids      BLOB NOT NULL,
            k_u            INTEGER NOT NULL DEFAULT 1,
            diversity      REAL NOT NULL DEFAULT 0.5,
            created_at     TEXT NOT NULL,
            last_active    TEXT NOT NULL,
            thread_weights BLOB DEFAULT NULL,
            thread_labels  TEXT DEFAULT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            arxiv_id   TEXT NOT NULL,
            signal     TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            score      REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_papers (
            user_id    TEXT NOT NULL,
            arxiv_id   TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, arxiv_id)
        )
    """)
    # Additive migration: add thread metadata columns if missing.
    for col in ("thread_weights BLOB DEFAULT NULL",
                "thread_labels TEXT DEFAULT NULL",
                "username TEXT",
                "password_hash TEXT"):
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # for research
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            arxiv_id   TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str) -> str:
    """Return a salted PBKDF2 hash in a compact string format."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return (
        f"pbkdf2_sha256$200000$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def _verify_password(password: str, encoded_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash."""
    try:
        algorithm, rounds_txt, salt_b64, digest_b64 = encoded_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        rounds = int(rounds_txt)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
    except (ValueError, TypeError):
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return secrets.compare_digest(candidate, expected)


def create_user(
    display_name: str,
    centroids: np.ndarray,
    k_u: int,
    diversity: float = 0.5,
    thread_weights: np.ndarray | None = None,
    thread_labels: list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Create a new user record in the database.

    Args:
        display_name: The user's display name.
        centroids: The user's initial centroids, shape (k_u, 768), float32, unit-norm rows.
        k_u: Number of research threads (1–3).
        diversity: The diversity slider value, 0.0–1.0.
        thread_weights: Optional (k_u,) float32 array of per-thread importance.
        thread_labels: Optional list of k_u human-readable thread names.
        username: Optional account username for returning-user login.
        password: Optional plaintext password (stored as PBKDF2 hash).

    Returns:
        The generated user_id (UUID string).
    """
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    blob = centroids.astype(np.float32).tobytes()
    tw_blob = thread_weights.astype(np.float32).tobytes() if thread_weights is not None else None
    tl_json = json.dumps(thread_labels) if thread_labels is not None else None
    normalized_username = username.strip().lower() if username else None
    password_hash = _hash_password(password) if password else None

    conn = _connect()
    if normalized_username:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
        if exists is not None:
            conn.close()
            raise ValueError("Username already exists.")

    conn.execute(
        "INSERT INTO users (user_id, display_name, username, password_hash, centroids, "
        "k_u, diversity, created_at, last_active, thread_weights, thread_labels) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            display_name,
            normalized_username,
            password_hash,
            blob,
            k_u,
            diversity,
            now,
            now,
            tw_blob,
            tl_json,
        ),
    )
    conn.commit()
    conn.close()
    return user_id


def _default_thread_weights(k_u: int) -> np.ndarray:
    if k_u <= 0:
        return np.array([], dtype=np.float32)
    return np.full(k_u, 1.0 / k_u, dtype=np.float32)


def _default_thread_labels(k_u: int) -> list[str]:
    return [f"Thread {i + 1}" for i in range(k_u)]


def get_user(user_id: str) -> dict | None:
    """Retrieve a user record by user_id.

    Args:
        user_id: The UUID string of the user.

    Returns:
        Dict with keys: user_id, display_name, centroids (np.ndarray shape (k_u, 768)),
        k_u, diversity, created_at, last_active, thread_weights, thread_labels.
        Returns None if user not found.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT user_id, display_name, username, centroids, k_u, diversity, "
        "created_at, last_active, thread_weights, thread_labels "
        "FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    k_u = row[4]
    tw_blob = row[8]
    tl_text = row[9]

    thread_weights = _default_thread_weights(k_u)
    if tw_blob is not None:
        loaded_weights = np.frombuffer(tw_blob, dtype=np.float32).copy()
        if loaded_weights.shape == (k_u,):
            thread_weights = loaded_weights

    thread_labels = _default_thread_labels(k_u)
    if tl_text is not None:
        try:
            loaded_labels = json.loads(tl_text)
        except json.JSONDecodeError:
            loaded_labels = None
        if (
            isinstance(loaded_labels, list)
            and len(loaded_labels) == k_u
            and all(isinstance(label, str) for label in loaded_labels)
        ):
            thread_labels = loaded_labels

    return {
        "user_id": row[0],
        "display_name": row[1],
        "username": row[2],
        "centroids": np.frombuffer(row[3], dtype=np.float32).copy().reshape(k_u, 768),
        "k_u": k_u,
        "diversity": row[5],
        "created_at": row[6],
        "last_active": row[7],
        "thread_weights": thread_weights,
        "thread_labels": thread_labels,
    }


def authenticate_user(username: str, password: str) -> dict | None:
    """Authenticate by username/password and return user payload on success."""
    normalized_username = username.strip().lower()
    conn = _connect()
    row = conn.execute(
        "SELECT user_id, password_hash FROM users WHERE username = ?",
        (normalized_username,),
    ).fetchone()
    conn.close()

    if row is None:
        return None
    user_id, password_hash = row
    if not password_hash or not _verify_password(password, password_hash):
        return None
    return get_user(user_id)


def update_centroids(user_id: str, centroids: np.ndarray) -> None:
    """Update a user's centroids in the database.

    Args:
        user_id: The UUID string of the user.
        centroids: The new centroids, shape (k_u, 768), float32, unit-norm rows.
    """
    now = datetime.now(timezone.utc).isoformat()
    blob = centroids.astype(np.float32).tobytes()

    conn = _connect()
    conn.execute(
        "UPDATE users SET centroids = ?, last_active = ? WHERE user_id = ?",
        (blob, now, user_id),
    )
    conn.commit()
    conn.close()


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
    """
    now = datetime.now(timezone.utc).isoformat()

    conn = _connect()
    conn.execute(
        "INSERT INTO feedback (user_id, arxiv_id, signal, cluster_id, score, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, arxiv_id, signal, cluster_id, score, now),
    )
    conn.commit()
    conn.close()


def get_seen_ids(user_id: str) -> set[str]:
    """Get the set of all arXiv paper IDs this user has been served or acted on.

    Args:
        user_id: The UUID string of the user.

    Returns:
        Set of arxiv_id strings from served daily feeds and feedback rows.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT arxiv_id FROM feedback WHERE user_id = ? "
        "UNION SELECT arxiv_id FROM seen_papers WHERE user_id = ?",
        (user_id, user_id),
    ).fetchall()
    conn.close()

    return {row[0] for row in rows}


def mark_papers_seen(
    user_id: str,
    arxiv_ids: list[str] | tuple[str, ...] | set[str],
) -> None:
    """Mark final served daily-feed papers as seen.

    This is intentionally separate from feedback so generated-but-not-served
    candidates never affect future retrieval, and served papers do not count as
    likes/saves/skips.
    """
    clean_ids = sorted({pid for pid in arxiv_ids if pid})
    if not clean_ids:
        return

    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    conn.executemany(
        "INSERT OR IGNORE INTO seen_papers (user_id, arxiv_id, created_at) "
        "VALUES (?, ?, ?)",
        [(user_id, arxiv_id, now) for arxiv_id in clean_ids],
    )
    conn.commit()
    conn.close()


def get_saved_papers(user_id: str) -> list[dict]:
    """Get all papers this user saved, newest first.

    Returns:
        List of dicts with keys: arxiv_id, score, created_at.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT arxiv_id, score, created_at FROM feedback "
        "WHERE user_id = ? AND signal = 'save' "
        "ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()

    return [
        {"arxiv_id": r[0], "score": r[1], "created_at": r[2]}
        for r in rows
    ]


def get_feedback_counts(user_id: str) -> dict[str, int]:
    """Get counts of each feedback type for a user.

    Returns:
        Dict with keys 'like', 'save', 'skip' mapped to counts.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT signal, COUNT(*) FROM feedback "
        "WHERE user_id = ? GROUP BY signal",
        (user_id,),
    ).fetchall()
    conn.close()

    counts = {"like": 0, "save": 0, "skip": 0}
    for signal, count in rows:
        counts[signal] = count
    return counts


def get_interacted_paper_count(user_id: str) -> int:
    """Count distinct papers this user has explicitly acted on."""
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(DISTINCT arxiv_id) FROM feedback WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return int(row[0] if row else 0)


# research mode
def save_research_note(user_id: str, content: str, source_arxiv_id: str = None):
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO research_notes (user_id, content, arxiv_id, created_at) VALUES (?, ?, ?, ?)",
        (user_id, content, source_arxiv_id, now)
    )
    conn.commit()
    conn.close()

def get_all_notes(user_id: str):
    conn = _connect()
    # Order by created_at so it feels like a chronological "wall"
    rows = conn.execute(
        "SELECT content, arxiv_id, created_at FROM research_notes WHERE user_id = ? ORDER BY created_at ASC",
        (user_id,)
    ).fetchall()
    conn.close()
    return rows

def delete_saved_paper(user_id: str, arxiv_id: str):
    """
    Removes a 'save' feedback signal for a specific paper.
    This effectively 'unsaves' the paper from the user's archive and folders.
    """
    conn = _connect()
    # We only delete the 'save' signal. 
    # If the user also 'liked' it, that remains in their profile history.
    conn.execute(
        "DELETE FROM feedback WHERE user_id = ? AND arxiv_id = ? AND signal = 'save'",
        (user_id, arxiv_id)
    )
    conn.commit()
    conn.close()