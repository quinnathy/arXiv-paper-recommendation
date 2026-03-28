# ArXiv Paper Recommendation Engine — Implementation Plan

> **For Claude Code:** Read this document fully before writing any code.
> Follow the phases in order. Each phase ends with a working, runnable state.
> Do not skip ahead. Do not add features not listed here.

---

## Project Overview

A Streamlit web app that recommends 3 ArXiv papers per day to a user, personalized by topic
selection (cold-start) and like/skip/save feedback (EMA-based embedding updates).

**Stack:**
- Python 3.11+
- Streamlit (UI + deployment)
- sentence-transformers (SPECTER2 embeddings)
- scikit-learn (MiniBatchKMeans)
- numpy (KNN, EMA, linear algebra)
- SQLite via Python `sqlite3` (user profiles, feedback log)
- kagglehub (dataset download)

**No Docker. No external vector DB. No cloud services required.**
Everything runs locally. SQLite file lives next to the app.

---

## Repository Layout

```
arxiv-rec/
├── IMPLEMENTATION_PLAN.md       ← this file
├── README.md
├── requirements.txt
├── .streamlit/
│   └── config.toml              ← theme + server config
│
├── app.py                       ← Streamlit entry point (thin shell)
│
├── pipeline/
│   ├── __init__.py
│   ├── embed.py                 ← SPECTER2 wrapper
│   ├── cluster.py               ← k-means + category centroids
│   ├── index.py                 ← numpy KNN search
│   └── offline.py               ← orchestrates embed → cluster → save
│
├── user/
│   ├── __init__.py
│   ├── db.py                    ← SQLite schema + CRUD
│   ├── profile.py               ← cold-start init, EMA update
│   └── session.py               ← Streamlit session state helpers
│
├── recommender/
│   ├── __init__.py
│   ├── retrieve.py              ← cluster selection + KNN
│   ├── rerank.py                ← recency boost + diversity filter
│   └── engine.py                ← top-level recommend() function
│
├── ui/
│   ├── __init__.py
│   ├── onboarding.py            ← topic selection page
│   ├── daily_feed.py            ← 3-paper card view
│   └── components.py            ← reusable Streamlit widgets
│
├── data/
│   ├── .gitkeep
│   ├── embeddings.npy           ← (generated, gitignored) N×768 float32
│   ├── cluster_ids.npy          ← (generated, gitignored) N, int32
│   ├── centroids.npy            ← (generated, gitignored) 500×768 float32
│   ├── category_centroids.npy   ← (generated, gitignored) dict serialized
│   ├── paper_meta.jsonl         ← (generated, gitignored) one JSON per line
│   └── arxiv_rec.db             ← (generated, gitignored) SQLite file
│
└── scripts/
    └── run_offline_pipeline.py  ← CLI: python scripts/run_offline_pipeline.py
```

---

## Phase 0 — Project Scaffold

**Goal:** Repo exists, dependencies install, `streamlit run app.py` shows a placeholder.

### 0.1 Create `requirements.txt`

```
streamlit>=1.35
numpy>=1.26
scikit-learn>=1.4
sentence-transformers>=3.0
kagglehub>=0.2
torch>=2.2          # CPU-only is fine; sentence-transformers dep
```

### 0.2 Create `.streamlit/config.toml`

```toml
[theme]
base = "light"
primaryColor = "#2a4d8f"
backgroundColor = "#f7f5f0"
secondaryBackgroundColor = "#ffffff"
font = "serif"

[server]
headless = true
port = 8501
```

### 0.3 Create `app.py` (placeholder)

```python
import streamlit as st

st.set_page_config(
    page_title="ArXiv Daily",
    page_icon="📄",
    layout="centered",
)

st.title("ArXiv Daily — coming soon")
```

### 0.4 Verify

```bash
pip install -r requirements.txt
streamlit run app.py
# Should show placeholder page at localhost:8501
```

---

## Phase 1 — Offline Pipeline

**Goal:** `python scripts/run_offline_pipeline.py` downloads data, embeds papers,
clusters, and writes all artifacts to `data/`. This runs once (or nightly via cron).

> **Important:** The pipeline processes up to 2M papers. On first run with a GPU this
> takes ~4 hours. On CPU only, plan for overnight. Use `--limit` flag during development.

### 1.1 `pipeline/embed.py`

Implement `EmbeddingModel` class:

```python
class EmbeddingModel:
    MODEL_NAME = "allenai/specter2_base"

    def __init__(self):
        # Load SentenceTransformer model
        # Set device to "cuda" if available, else "cpu"

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        # Encode texts with batch_size=64, normalize_embeddings=True
        # Returns float32 array shape (len(texts), 768)

    def embed_papers(self, papers: list[dict]) -> np.ndarray:
        # papers is list of {"title": str, "abstract": str}
        # Concatenate as "title. abstract" for each paper
        # Call embed_batch, return result
```

**Key detail:** Always pass `normalize_embeddings=True` to `encode()`. This makes
every vector unit-length, so cosine similarity = dot product everywhere downstream.

### 1.2 `pipeline/cluster.py`

Implement two functions:

```python
def fit_kmeans(embeddings: np.ndarray, k: int = 500) -> tuple[np.ndarray, np.ndarray]:
    # Use MiniBatchKMeans(n_clusters=k, n_init=5, random_state=42, batch_size=4096)
    # Returns (cluster_ids: shape N, centroids: shape k×768)
    # Normalize centroids to unit length before returning

def compute_category_centroids(
    embeddings: np.ndarray,
    paper_meta: list[dict],
) -> dict[str, np.ndarray]:
    # paper_meta[i]["categories"] is a list of arXiv tags e.g. ["cs.LG", "cs.CL"]
    # For each unique tag, compute mean of all embeddings where that tag appears
    # Normalize each mean to unit length
    # Returns dict: {"cs.LG": array(768,), "cs.CL": array(768,), ...}
```

**Note:** `compute_category_centroids` is independent of k-means. Run it from the
same embeddings. Store separately as `data/category_centroids.npy` using
`np.save(..., allow_pickle=True)` since the value is a dict.

### 1.3 `pipeline/offline.py`

Orchestrate the full pipeline:

```python
def run(limit: int | None = None, k: int = 500, data_dir: str = "data"):
    # Step 1: Download dataset with kagglehub
    #   import kagglehub
    #   path = kagglehub.dataset_download("Cornell-University/arxiv")
    #   Find the .json file inside path

    # Step 2: Load papers from JSON (one record per line)
    #   Fields to keep: id, title, abstract, categories, update_date
    #   If limit is set, stop after limit records
    #   Skip records missing title or abstract

    # Step 3: Embed
    #   model = EmbeddingModel()
    #   embeddings = model.embed_papers(papers)

    # Step 4: Cluster + category centroids (run in parallel, both use same embeddings)
    #   cluster_ids, centroids = fit_kmeans(embeddings, k)
    #   category_centroids = compute_category_centroids(embeddings, papers)

    # Step 5: Attach cluster_id to each paper's metadata
    #   paper_meta[i]["cluster_id"] = int(cluster_ids[i])

    # Step 6: Save all artifacts to data_dir
    #   np.save("data/embeddings.npy", embeddings)
    #   np.save("data/cluster_ids.npy", cluster_ids)
    #   np.save("data/centroids.npy", centroids)
    #   np.save("data/category_centroids.npy", category_centroids, allow_pickle=True)
    #   Write paper_meta as JSONL to "data/paper_meta.jsonl"

    # Print progress at each step with timing
```

### 1.4 `scripts/run_offline_pipeline.py`

```python
import argparse
from pipeline.offline import run

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Max papers to process (None = all). Use 10000 for dev.")
parser.add_argument("--k", type=int, default=500)
args = parser.parse_args()

run(limit=args.limit, k=args.k)
```

### 1.5 Development shortcut

During development, always use:
```bash
python scripts/run_offline_pipeline.py --limit 50000
```
50K papers is enough to test the full pipeline in ~5 minutes on CPU. Switch to full
dataset only when the app logic is complete.

### 1.6 Verify Phase 1

```bash
python scripts/run_offline_pipeline.py --limit 10000
# Should produce: data/embeddings.npy, cluster_ids.npy, centroids.npy,
#                 category_centroids.npy, paper_meta.jsonl
# Check shapes: embeddings (10000, 768), centroids (500, 768)
```

---

## Phase 2 — User Database

**Goal:** SQLite schema is created automatically on first run. User profiles persist
across Streamlit sessions.

### 2.1 `user/db.py`

Schema — two tables:

**Table: `users`**
| Column | Type | Notes |
|--------|------|-------|
| user_id | TEXT PK | UUID string |
| display_name | TEXT | |
| embedding | BLOB | numpy array, stored as raw bytes via `array.tobytes()` |
| created_at | TEXT | ISO datetime |
| last_active | TEXT | ISO datetime |

**Table: `feedback`**
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK AUTOINCREMENT | |
| user_id | TEXT FK | |
| arxiv_id | TEXT | |
| signal | TEXT | "like" \| "save" \| "skip" |
| cluster_id | INTEGER | for analytics |
| score | REAL | recommendation score at time of serving |
| created_at | TEXT | ISO datetime |

Implement these functions (no ORM, plain `sqlite3`):

```python
def init_db(db_path: str = "data/arxiv_rec.db") -> None:
    # Create tables if not exist
    # Call this at app startup

def create_user(display_name: str, embedding: np.ndarray) -> str:
    # Generate UUID, insert row, return user_id

def get_user(user_id: str) -> dict | None:
    # Return dict with all fields; decode embedding bytes → np.ndarray float32

def update_embedding(user_id: str, embedding: np.ndarray) -> None:
    # UPDATE users SET embedding=? WHERE user_id=?

def log_feedback(user_id: str, arxiv_id: str, signal: str,
                 cluster_id: int, score: float) -> None:
    # INSERT into feedback

def get_seen_ids(user_id: str) -> set[str]:
    # SELECT arxiv_id FROM feedback WHERE user_id=?
    # Returns set of all arxiv_ids this user has seen (any signal)
```

**Embedding serialization:**
```python
# Store:  embedding.astype(np.float32).tobytes()
# Load:   np.frombuffer(blob, dtype=np.float32)
```

### 2.2 `user/profile.py`

```python
FEEDBACK_WEIGHTS = {"like": 1.0, "save": 1.5, "skip": -0.3}
EMA_ALPHA = 0.15

def init_embedding_from_topics(
    selected_categories: list[str],
    category_centroids: dict[str, np.ndarray],
) -> np.ndarray:
    # Average the centroids for selected categories
    # Normalize to unit length
    # If no match found, return the first centroid as fallback
    # Returns shape (768,) float32

def apply_feedback(
    user_embedding: np.ndarray,
    paper_embedding: np.ndarray,
    signal: str,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    # w = FEEDBACK_WEIGHTS[signal]
    # raw = (1 - alpha) * user_embedding + alpha * w * paper_embedding
    # norm = np.linalg.norm(raw)
    # if norm < 1e-8: return user_embedding  (guard)
    # return raw / norm
```

### 2.3 `user/session.py`

Streamlit-specific helpers that bridge `st.session_state` and the database:

```python
def load_or_init_session(db_path: str) -> None:
    # If "user_id" not in st.session_state:
    #   Set st.session_state["user_id"] = None
    #   Set st.session_state["user_embedding"] = None
    #   Set st.session_state["onboarded"] = False

def login_user(user_id: str, db_path: str) -> bool:
    # Load user from DB, populate session_state
    # Return True if found, False otherwise

def save_embedding_to_session(embedding: np.ndarray) -> None:
    # st.session_state["user_embedding"] = embedding

def is_onboarded() -> bool:
    # Return st.session_state.get("onboarded", False)
```

### 2.4 Verify Phase 2

Write a small test in `scripts/test_db.py`:
```python
from user.db import init_db, create_user, get_user, log_feedback, get_seen_ids
import numpy as np

init_db()
emb = np.random.randn(768).astype(np.float32)
emb /= np.linalg.norm(emb)
uid = create_user("Test User", emb)
print("Created:", uid)
user = get_user(uid)
print("Retrieved embedding shape:", user["embedding"].shape)
log_feedback(uid, "2401.00001", "like", cluster_id=3, score=0.87)
print("Seen IDs:", get_seen_ids(uid))
```

---

## Phase 3 — Recommendation Engine

**Goal:** `engine.recommend(user_embedding, seen_ids)` returns 3 paper dicts.

### 3.1 `pipeline/index.py`

```python
class PaperIndex:
    """
    Holds all data needed for serving. Loaded once at startup into memory.
    """
    def __init__(self, data_dir: str = "data"):
        self.embeddings: np.ndarray       # shape (N, 768) float32
        self.cluster_ids: np.ndarray      # shape (N,) int32
        self.centroids: np.ndarray        # shape (k, 768) float32
        self.category_centroids: dict     # str → np.ndarray(768)
        self.paper_meta: list[dict]       # length N

    def load(self) -> None:
        # Load all artifacts from data_dir
        # Verify shapes are consistent (len(embeddings) == len(paper_meta))
        # Print memory usage: embeddings.nbytes / 1e9, "GB"

    def is_loaded(self) -> bool:
        return self.embeddings is not None
```

**Memory note:** At N=2M, embeddings.npy is 5.9 GB. At N=50K (dev), it's ~147 MB.
The index is loaded once and cached in `st.cache_resource`.

### 3.2 `recommender/retrieve.py`

```python
def find_nearest_clusters(
    user_emb: np.ndarray,
    centroids: np.ndarray,
    n: int = 2,
) -> list[int]:
    # sims = centroids @ user_emb   ← shape (k,)
    # return top-n indices by descending similarity

def knn_in_clusters(
    user_emb: np.ndarray,
    target_cluster_ids: list[int],
    index: PaperIndex,
    seen_ids: set[str],
    k: int = 40,
) -> list[tuple[float, dict]]:
    # mask = np.isin(index.cluster_ids, target_cluster_ids)
    # cand_embs = index.embeddings[mask]       ← shape (M, 768)
    # cand_meta = [index.paper_meta[i] for i, b in enumerate(mask) if b]
    # sims = cand_embs @ user_emb              ← shape (M,)
    # Sort descending, filter seen_ids, return top k as [(score, meta), ...]
```

### 3.3 `recommender/rerank.py`

```python
def recency_score(published_date: str, halflife_days: float = 30.0) -> float:
    # Parse published_date (ISO format from paper_meta)
    # age_days = (datetime.now() - published).days
    # return exp(-age_days / halflife_days)
    # Clamp age to max 365 days to avoid near-zero scores on old papers

def rerank_and_select(
    candidates: list[tuple[float, dict]],
    recency_weight: float = 0.25,
    n: int = 3,
) -> list[dict]:
    # For each (sim_score, meta) in candidates:
    #   score = sim_score + recency_weight * recency_score(meta["update_date"])
    #   attach score to meta dict as meta["rec_score"]
    # Sort by score descending
    # Diversity pass: iterate sorted; add paper only if cluster_id not yet used
    # Stop when n papers selected
    # Return list of n meta dicts
```

### 3.4 `recommender/engine.py`

```python
def recommend(
    user_emb: np.ndarray,
    seen_ids: set[str],
    index: PaperIndex,
    n: int = 3,
) -> list[dict]:
    # 1. find_nearest_clusters(user_emb, index.centroids, n=2)
    # 2. knn_in_clusters(user_emb, clusters, index, seen_ids, k=40)
    # 3. rerank_and_select(candidates, n=n)
    # 4. return result
    #
    # Edge case: if fewer than n papers returned (new user, small index),
    # fall back to sampling from all clusters, not just top-2
```

### 3.5 Verify Phase 3

```python
# scripts/test_engine.py
from pipeline.index import PaperIndex
from recommender.engine import recommend
import numpy as np

index = PaperIndex()
index.load()

# fake user embedding
user_emb = np.random.randn(768).astype(np.float32)
user_emb /= np.linalg.norm(user_emb)

recs = recommend(user_emb, seen_ids=set(), index=index)
for i, r in enumerate(recs):
    print(f"{i+1}. [{r['id']}] {r['title'][:60]}  score={r['rec_score']:.3f}")
```

---

## Phase 4 — Streamlit UI

**Goal:** Full working app. Three views: onboarding → daily feed → feedback.

### 4.1 `ui/components.py`

Reusable widgets:

```python
def paper_card(meta: dict, on_like, on_save, on_skip) -> None:
    """
    Renders one paper as a Streamlit card:
    - Title (st.subheader)
    - Category badges (st.caption with colored tags)
    - Abstract snippet: first 300 characters + "..."
    - Links: [ArXiv page] [PDF]
    - Three action buttons: Like | Save | Skip
    Calls on_like/on_save/on_skip(arxiv_id) when clicked.
    """

def topic_selector(category_centroids: dict) -> list[str]:
    """
    Multiselect widget showing human-readable topic labels.
    Maps labels to arXiv category codes.
    Returns list of selected arXiv category strings.

    Mapping (hardcode this dict):
    {
      "Machine Learning": "cs.LG",
      "Computer Vision": "cs.CV",
      "Natural Language Processing": "cs.CL",
      "Robotics": "cs.RO",
      "Statistics / ML Theory": "stat.ML",
      "Artificial Intelligence": "cs.AI",
      "Computation & Language": "cs.CL",
      "Neural Networks": "cs.NE",
      "Information Retrieval": "cs.IR",
      "Human-Computer Interaction": "cs.HC",
      "Cryptography & Security": "cs.CR",
      "Distributed Computing": "cs.DC",
      "Computational Biology": "q-bio.QM",
      "Physics & ML": "physics.comp-ph",
      "Quantitative Finance": "q-fin.CP",
    }
    Only show topics that exist in category_centroids (i.e., appear in the corpus).
    """

def loading_spinner_with_message(message: str):
    """Context manager wrapping st.spinner with a custom message."""
```

### 4.2 `ui/onboarding.py`

```python
def render_onboarding(index: PaperIndex, db_path: str) -> None:
    """
    Page shown to new users (not yet onboarded).

    Layout:
    1. App title + one-line description
    2. Name input field (st.text_input)
    3. topic_selector() widget
    4. "Start reading" button

    On button click:
    - Validate: name non-empty, at least 1 topic selected
    - embedding = init_embedding_from_topics(selected, index.category_centroids)
    - user_id = create_user(name, embedding)
    - Update st.session_state: user_id, user_embedding, onboarded=True
    - st.rerun() to trigger main feed
    """
```

### 4.3 `ui/daily_feed.py`

```python
def render_daily_feed(index: PaperIndex, db_path: str) -> None:
    """
    Main page for onboarded users.

    Layout:
    1. Header: "Good morning, {name}" + today's date
    2. Subheader: "Your 3 papers for today"
    3. If today's recommendations not yet generated:
       - Call recommend() with current user_embedding + seen_ids
       - Store results in st.session_state["todays_recs"]
    4. Render three paper_card() widgets
    5. Sidebar:
       - User name + "since {created_at}"
       - "Liked papers: N" counter
       - "Settings" expander with topic re-selection

    Feedback handling (on_like/on_save/on_skip callbacks):
    - log_feedback() to DB
    - apply_feedback() to update embedding in session_state
    - update_embedding() to persist to DB
    - Mark paper as "responded" in session state (disable buttons)
    - If all 3 responded: show "Come back tomorrow!" message
    """
```

### 4.4 `app.py` (final version)

```python
import streamlit as st
from pipeline.index import PaperIndex
from user.db import init_db
from user.session import load_or_init_session, is_onboarded
from ui.onboarding import render_onboarding
from ui.daily_feed import render_daily_feed

DB_PATH = "data/arxiv_rec.db"

st.set_page_config(
    page_title="ArXiv Daily",
    page_icon="📄",
    layout="centered",
)

@st.cache_resource
def load_index() -> PaperIndex:
    idx = PaperIndex()
    idx.load()
    return idx

# ── init ──────────────────────────────────────────
init_db(DB_PATH)
load_or_init_session(DB_PATH)
index = load_index()

# ── check if pipeline has run ──────────────────────
if not index.is_loaded():
    st.error("Data not found. Run: python scripts/run_offline_pipeline.py --limit 50000")
    st.stop()

# ── route ─────────────────────────────────────────
if not is_onboarded():
    render_onboarding(index, DB_PATH)
else:
    render_daily_feed(index, DB_PATH)
```

### 4.5 Verify Phase 4

```bash
streamlit run app.py
# Visit localhost:8501
# Complete onboarding → see 3 paper cards
# Click Like on paper 1 → embedding updates → DB updated
# Refresh page → same 3 papers shown (from session state cache)
```

---

## Phase 5 — Polish and Edge Cases

Handle these before considering the app done:

### 5.1 Data not ready guard

In `app.py`: if `data/embeddings.npy` does not exist, show a clear error with the
exact command to run. Do not crash.

### 5.2 Cold-start fallback

In `init_embedding_from_topics`: if selected_categories has no overlap with
category_centroids (unlikely but possible), fall back to `centroids[0]` and log a warning.

### 5.3 Recommendation exhaustion

If `recommend()` returns fewer than 3 papers (user has seen almost everything, or
index is tiny in dev mode), show however many are available with a note:
"You've seen most papers in your areas — come back tomorrow for new ones."

### 5.4 Feedback idempotency

In the UI, once a user clicks Like/Save/Skip on a paper, disable all three buttons
for that card. Store responded paper IDs in `st.session_state["responded"]` (a set).
Do not allow double-logging.

### 5.5 Daily reset

Papers are not re-served once seen (they are in `seen_ids` forever). The "daily"
mechanic works naturally: new papers arrive in the index nightly via the cron script,
and old papers are excluded by `seen_ids`. No date-based logic needed.

---

## Implementation Order Summary

| Phase | What to build | Runnable check |
|-------|--------------|----------------|
| 0 | Scaffold + placeholder | `streamlit run app.py` |
| 1 | Offline pipeline | `python scripts/run_offline_pipeline.py --limit 10000` |
| 2 | SQLite user DB | `python scripts/test_db.py` |
| 3 | Recommendation engine | `python scripts/test_engine.py` |
| 4 | Streamlit UI | Full app in browser |
| 5 | Polish + edge cases | Manual QA |

---

## Key Invariants — Never Violate These

1. **All embeddings are unit-norm.** Normalize immediately after encoding. Normalize
   after every EMA update. If any vector has `np.linalg.norm(v) < 0.99`, something is wrong.

2. **Dot product = cosine similarity.** Because all vectors are unit-norm, `A @ b`
   gives cosine similarity. Never compute cosine with a division — if you find yourself
   dividing, it means vectors weren't normalized.

3. **`seen_ids` is cumulative.** Once a paper is served (regardless of signal), it
   goes into `seen_ids` and is never served again. Log feedback separately.

4. **EMA update only on explicit feedback.** Do not call `apply_feedback` if the user
   did not interact. Non-interaction ≠ skip.

5. **`st.cache_resource` for the index.** The 6 GB embedding matrix must be loaded
   exactly once per process. Never load it inside a function that can be called
   repeatedly by Streamlit's re-run model.

6. **SQLite writes are synchronous.** `user/db.py` uses no threading. Streamlit runs
   single-user per session; this is fine.

---

## Notes for Claude Code

- Write docstrings for every public function.
- Use type hints throughout.
- Print progress to stdout in the offline pipeline (not logging — `print()` is fine here).
- In Streamlit code, prefer `st.session_state` over module-level globals.
- When in doubt about a design decision, follow the algorithm description document
  (`IMPLEMENTATION_PLAN.md` §§ 1–3) rather than improvising.
- Do not add caching layers beyond `st.cache_resource` for the index.
- Do not use async/await — Streamlit's threading model does not need it.
