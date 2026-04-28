# ArXiv Daily

A personalized paper recommendation engine that surfaces 5 arXiv papers daily, tailored to your research interests. Built with Streamlit, SPECTER2 embeddings, and a lightweight local stack — no cloud services or external databases required.

## How It Works

### The Core Idea

Your research taste is represented as **multiple 768-dimensional vectors** (research threads) in the same embedding space as every paper in the arXiv corpus (~2M papers). Recommendation is geometric: the system finds papers whose embeddings are closest to any of your research threads. A diversity slider (delta) controls how broadly recommendations spread across your interests.

### Algorithm Overview

The system has two phases: an **offline batch pipeline** that runs once (or nightly), and an **online serving pipeline** that executes per user at delivery time.

#### 1. Offline Pipeline

1. **Corpus preparation** — The arXiv dataset is loaded from its [Kaggle JSON snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv). Each record provides an ID, title, abstract, category tags (e.g. `cs.LG`), and date.
2. **Embedding** — [SPECTER2](https://huggingface.co/allenai/specter2_base), a scientific-text encoder pretrained on citation graphs, encodes each paper's *title + abstract* into a 768-dim vector. All vectors are L2-normalized to unit length, so cosine similarity reduces to a simple dot product.
3. **Clustering** — MiniBatchKMeans partitions papers into 500 clusters. This acts as a spatial index: instead of scanning all 2M papers at query time, only papers in the nearest clusters are searched.
4. **Category centroids** — Independently, one centroid per arXiv category label (e.g. `cs.LG`, `cs.CV`) is computed as the normalized mean of all embeddings in that category. These are used for cold-start user initialization.

#### 2. User Personalization

- **Cold start (multi-vector)** — A new user selects topics of interest. Optionally, they can paste a Google Scholar profile URL; the system fetches their publications and embeds them via SPECTER2. All seed vectors (category centroids + optional paper embeddings) are clustered with KMeans into `k_u` centroids (1-3), where `k_u = min(3, number of selected topics)`. Each centroid represents a distinct research thread.
- **Diversity slider (delta)** — During onboarding the user sets delta (0.0-1.0), which controls how broadly recommendations spread. Lower values focus on the user's strongest interest; higher values explore more broadly.
- **Feedback loop** — Each like, save, or skip updates only the **nearest centroid** via Exponential Moving Average (EMA):

  ```
  i* = argmax(centroids @ e_paper)
  centroids[i*] = normalize( (1 - 0.15) * centroids[i*] + 0.15 * w * e_paper )
  ```

  where `w` = +1.5 (save), +1.0 (like), or -0.3 (skip). Only the closest research thread moves; all others remain unchanged.

#### 3. Online Serving

1. **Cluster selection** — The total cluster budget is `ceil(2 + delta * 3)` (delta=0 gives 2 clusters, delta=1 gives 5). Budget is split evenly across the user's `k_u` centroids. Each centroid selects its top clusters by dot product. Results are deduplicated.
2. **KNN retrieval** — Brute-force dot-product search within those clusters. Each paper is scored against **all** user centroids; the maximum similarity (nearest research thread) is used. Filtering out previously seen papers, the top 40 candidates are kept.
3. **Re-ranking** — Adjust scores with a recency bonus: `score = similarity + 0.25 * exp(-age_days / 30)`. Newer papers get a boost.
4. **Diversity filter** — At most one paper per k-means cluster is selected (always enforced). When delta > 0.5 and k_u > 1, output slots are held open for uncovered user centroids before filling by score alone. Stop at 5 papers.

Total serving cost: <1 ms on CPU per user.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | ~2,000,000 | Papers in the corpus |
| d | 768 | Embedding dimension (SPECTER2) |
| k | 500 | Number of k-means clusters |
| k_u | 1-3 | User research threads (centroids) |
| delta | 0.0-1.0 | Diversity slider |
| alpha | 0.15 | EMA learning rate |
| M | 40 | KNN candidate pool size |
| n | 5 | Papers served per day |

## Tech Stack

- **Python 3.11+**
- **Streamlit** — UI and app framework
- **SPECTER2** (transformers + adapters) — Scientific paper embeddings
- **scikit-learn** — MiniBatchKMeans clustering, KMeans user profile initialization
- **NumPy** — KNN search, EMA updates, all linear algebra
- **SQLite** — User profiles and feedback log (via Python `sqlite3`)
- **kagglehub** — Dataset download
- **BeautifulSoup4** — Google Scholar profile scraping
- **requests** — HTTP client for Scholar and Semantic Scholar API

No Docker. No external vector DB. No cloud services. Everything runs locally with a SQLite file next to the app.

## Project Structure

```
arXiv-paper-recommendation/
├── app.py                       # Streamlit entry point
├── README.md
├── requirements.txt
├── .streamlit/config.toml       # Theme + server config
│
├── pipeline/
│   ├── __init__.py
│   ├── embed.py                 # SPECTER2 embedding wrapper
│   ├── cluster.py               # k-means + category centroids
│   ├── index.py                 # In-memory paper index for serving
│   ├── offline.py               # Orchestrates embed -> cluster -> save
│   ├── scholar_parser.py        # Google Scholar profile parser
│   └── runtime.py               # Single-thread runtime guards for NumPy/PyTorch
│
├── user/
│   ├── __init__.py
│   ├── db.py                    # SQLite schema + CRUD (multi-vector)
│   ├── profile.py               # Multi-centroid init, nearest-centroid EMA
│   └── session.py               # Streamlit session state helpers
│
├── recommender/
│   ├── __init__.py
│   ├── retrieve.py              # delta-aware cluster selection + multi-vector KNN
│   ├── rerank.py                # Recency boost + delta-aware diversity filter
│   └── engine.py                # Top-level recommend() function
│
├── ui/
│   ├── __init__.py
│   ├── components.py            # Reusable Streamlit widgets
│   ├── onboarding.py            # Topic selection + Scholar import + delta slider
│   └── daily_feed.py            # 5-paper card view + demo recommend button
│
├── data/                        # Local artifacts (created at runtime)
│   ├── .gitkeep
│   ├── embeddings.npy           # N x 768 float32
│   ├── cluster_ids.npy          # N, int32
│   ├── centroids.npy            # 500 x 768 float32
│   ├── category_centroids.npy   # Dict of category -> vector
│   ├── paper_meta.jsonl         # One JSON object per line
│   └── arxiv_rec.db             # SQLite user database
│
└── scripts/
    ├── run_offline_pipeline.py  # CLI for offline pipeline generation
    ├── reset_db.py              # Reset local SQLite test data
    ├── test_db.py               # Quick DB sanity checks
    └── test_engine.py           # Quick recommendation sanity checks
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the offline pipeline

This downloads the full arXiv dataset first, then samples/filters papers, embeds them, and clusters them. Use `--limit` for faster dev iterations:

```bash
# Development (fast, ~5 min on CPU)
python scripts/run_offline_pipeline.py --limit 50000

# Development with deterministic random sampling (same seed => same paper set)
python scripts/run_offline_pipeline.py --limit 50000 --seed 42

# Category-constrained sampling (supports top-level categories like cs/math/qfin)
python scripts/run_offline_pipeline.py --limit 50000 --categories cs,math,qfin --seed 42

# Category-constrained sampling by sub-categories (also supported)
python scripts/run_offline_pipeline.py --limit 50000 --categories cs.LG,cs.CV --seed 42

# Optional: use a CN Hugging Face mirror (default remains huggingface.co)
python scripts/run_offline_pipeline.py --limit 50000 --hf-endpoint https://hf-mirror.com --disable-hf-transfer

# Full dataset (~4 hours on GPU, overnight on CPU)
python scripts/run_offline_pipeline.py
```

### 3. Launch the app

```bash
streamlit run app.py
```

Visit `http://localhost:8501`, pick your topics, optionally paste a Google Scholar URL, set your diversity preference, and start reading.

### Reset test data (for testing use)

To clear local user/test data and recreate an empty DB schema:

```bash
python scripts/reset_db.py
```
