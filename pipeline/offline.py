"""Offline pipeline orchestrator: download → embed → cluster → save.

Runs the full data preparation pipeline. Meant to be invoked via
scripts/run_offline_pipeline.py (CLI) or programmatically.
Produces all artifacts in data_dir that the serving layer (PaperIndex) needs.
"""

from __future__ import annotations


def run(limit: int | None = None, k: int = 500, data_dir: str = "data") -> None:
    """Execute the full offline pipeline.

    Args:
        limit: Maximum number of papers to process. None = all (~2M).
            Use 10000–50000 during development for faster iteration.
        k: Number of k-means clusters. Default 500.
        data_dir: Output directory for all generated artifacts.

    Steps (print progress with timing at each step):

        Step 1 — Download dataset:
            - Use kagglehub.dataset_download("Cornell-University/arxiv").
            - Find the .json file inside the downloaded path.

        Step 2 — Load papers from JSON (one JSON object per line):
            - Fields to keep per paper: id, title, abstract, categories, update_date.
            - Parse categories from space-separated string into a list.
            - Skip records missing title or abstract.
            - If limit is set, stop after limit records.

        Step 3 — Embed papers:
            - Instantiate EmbeddingModel().
            - Call model.embed_papers(papers) to get (N, 768) float32 array.

        Step 4 — Cluster + category centroids:
            - cluster_ids, centroids = fit_kmeans(embeddings, k)
            - category_centroids = compute_category_centroids(embeddings, papers)

        Step 5 — Attach cluster_id to each paper's metadata:
            - papers[i]["cluster_id"] = int(cluster_ids[i])

        Step 6 — Save all artifacts to data_dir:
            - np.save(data_dir/embeddings.npy, embeddings)
            - np.save(data_dir/cluster_ids.npy, cluster_ids)
            - np.save(data_dir/centroids.npy, centroids)
            - np.save(data_dir/category_centroids.npy, category_centroids,
                      allow_pickle=True)
            - Write papers as JSONL to data_dir/paper_meta.jsonl
              (one JSON object per line).
    """
    raise NotImplementedError
