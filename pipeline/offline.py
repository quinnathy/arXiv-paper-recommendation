"""Offline pipeline orchestrator: download → embed → cluster → save.

Runs the full data preparation pipeline. Meant to be invoked via
scripts/run_offline_pipeline.py (CLI) or programmatically.
Produces all artifacts in data_dir that the serving layer (PaperIndex) needs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from random import Random

import kagglehub
import numpy as np

from pipeline.embed import EmbeddingModel
from pipeline.cluster import fit_kmeans, compute_category_centroids


def _normalize_requested_category(category: str) -> str:
    normalized = category.strip().lower()
    if normalized == "qfin":
        return "q-fin"
    return normalized


def _paper_matches_requested_category(
    paper_categories: list[str], requested_category: str
) -> bool:
    requested = _normalize_requested_category(requested_category)
    for paper_category in paper_categories:
        current = paper_category.lower()
        if current == requested or current.startswith(f"{requested}."):
            return True
    return False


def _sample_papers(
    papers: list[dict],
    limit: int | None,
    seed: int,
    categories: list[str] | None,
) -> list[dict]:
    if categories is not None and limit is None:
        return [
            paper
            for paper in papers
            if any(_paper_matches_requested_category(paper["categories"], c) for c in categories)
        ]

    if limit is None:
        return papers

    if limit <= 0:
        raise ValueError("limit must be > 0 when provided.")

    rng = Random(seed)
    if categories is None:
        if limit > len(papers):
            raise ValueError(f"limit={limit} exceeds available papers={len(papers)}.")
        return rng.sample(papers, k=limit)

    category_to_indices: dict[str, list[int]] = {c: [] for c in categories}
    for idx, paper in enumerate(papers):
        for c in categories:
            if _paper_matches_requested_category(paper["categories"], c):
                category_to_indices[c].append(idx)

    category_counts = [len(category_to_indices[c]) for c in categories]
    total_category_count = sum(category_counts)
    if total_category_count <= 0:
        raise ValueError("No papers found for requested categories.")
    normalized_ratios = [count / total_category_count for count in category_counts]

    available_indices: set[int] = set()
    for idxs in category_to_indices.values():
        available_indices.update(idxs)

    if limit > len(available_indices):
        raise ValueError(
            f"limit={limit} exceeds available papers in selected categories={len(available_indices)}."
        )

    targets = [int(limit * ratio) for ratio in normalized_ratios]
    remainder = limit - sum(targets)
    if remainder > 0:
        ordering = sorted(
            range(len(categories)),
            key=lambda i: (limit * normalized_ratios[i] - targets[i]),
            reverse=True,
        )
        for i in ordering[:remainder]:
            targets[i] += 1

    chosen_indices: set[int] = set()
    for i, category in enumerate(categories):
        pool = [idx for idx in category_to_indices[category] if idx not in chosen_indices]
        rng.shuffle(pool)
        take = min(targets[i], len(pool))
        chosen_indices.update(pool[:take])

    if len(chosen_indices) < limit:
        remaining_pool = list(available_indices - chosen_indices)
        rng.shuffle(remaining_pool)
        needed = limit - len(chosen_indices)
        chosen_indices.update(remaining_pool[:needed])

    chosen_list = list(chosen_indices)
    rng.shuffle(chosen_list)
    return [papers[idx] for idx in chosen_list]


def run(
    limit: int | None = None,
    k: int = 500,
    data_dir: str = "data",
    seed: int = 42,
    categories: list[str] | None = None,
) -> None:
    """Execute the full offline pipeline.

    Args:
        limit: Maximum number of papers to process. None = all (~2M).
            Use 10000-50000 during development for faster iteration.
        k: Number of k-means clusters. Default 500.
        data_dir: Output directory for all generated artifacts.
        seed: Random seed for reproducible sampling.
        categories: Optional category filters (e.g. ["cs.LG", "cs.CV"]).
            When set with limit, sampling only draws papers that include at least
            one of these categories. Supports both top-level categories (e.g.
            "cs", "math", "q-fin"/"qfin") and exact sub-categories (e.g. "cs.LG").
    """
    os.makedirs(data_dir, exist_ok=True)
    if categories:
        deduped: list[str] = []
        seen: set[str] = set()
        for category in categories:
            key = _normalize_requested_category(category)
            if key not in seen:
                seen.add(key)
                deduped.append(category)
        categories = deduped

    # ── Step 1: Download dataset ─────────────────────────────────────────
    t0 = time.time()
    print("Step 1/6: Downloading ArXiv dataset...")
    dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
    # Find the .json file inside the downloaded path
    json_file = None
    for root, _dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".json"):
                json_file = os.path.join(root, f)
                break
        if json_file:
            break
    if json_file is None:
        raise FileNotFoundError(f"No .json file found in {dataset_path}")
    print(f"  Dataset at: {json_file}  ({time.time() - t0:.1f}s)")

    # ── Step 2: Load papers from JSON ────────────────────────────────────
    t1 = time.time()
    print("Step 2/6: Loading papers from JSON...")
    all_papers: list[dict] = []
    with open(json_file, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            title = " ".join(record.get("title", "").split())
            abstract = record.get("abstract", "").strip()
            if not title or not abstract:
                continue
            all_papers.append({
                "id": record["id"],
                "title": title,
                "abstract": abstract,
                "categories": record.get("categories", "").split(),
                "update_date": record.get("update_date", ""),
            })
    print(f"  Loaded {len(all_papers)} valid papers  ({time.time() - t1:.1f}s)")

    print("  Sampling papers...")
    papers = _sample_papers(
        papers=all_papers,
        limit=limit,
        seed=seed,
        categories=categories,
    )
    if categories:
        category_to_count = {
            category: sum(
                1
                for p in all_papers
                if _paper_matches_requested_category(p["categories"], category)
            )
            for category in categories
        }
        total_count = sum(category_to_count.values())
        ratios_text = ",".join(
            f"{category}:{(category_to_count[category] / total_count):.4f}"
            for category in categories
        )
        print(
            "  Category filter enabled:"
            f" categories={','.join(categories)} auto-ratios={ratios_text}"
        )
    print(f"  Selected {len(papers)} papers with seed={seed}")

    # ── Step 3: Embed ────────────────────────────────────────────────────
    t2 = time.time()
    print("Step 3/6: Embedding papers with SPECTER2...")
    model = EmbeddingModel()
    embeddings = model.embed_papers(papers)
    print(f"  Embeddings shape: {embeddings.shape}  ({time.time() - t2:.1f}s)")

    # ── Step 4: Cluster + category centroids ─────────────────────────────
    t3 = time.time()
    print("Step 4/6: Clustering...")
    cluster_ids, centroids = fit_kmeans(embeddings, k)
    print(f"  Clusters: {centroids.shape[0]}, centroids shape: {centroids.shape}")

    print("  Computing category centroids...")
    category_centroids = compute_category_centroids(embeddings, papers)
    print(f"  Category centroids: {len(category_centroids)} categories  ({time.time() - t3:.1f}s)")

    # ── Step 5: Attach cluster_id to metadata ────────────────────────────
    t4 = time.time()
    print("Step 5/6: Attaching cluster IDs to metadata...")
    for i, cid in enumerate(cluster_ids):
        papers[i]["cluster_id"] = int(cid)
    print(f"  Done  ({time.time() - t4:.1f}s)")

    # ── Step 6: Save artifacts ───────────────────────────────────────────
    t5 = time.time()
    print("Step 6/6: Saving artifacts...")
    data = Path(data_dir)

    np.save(data / "embeddings.npy", embeddings)
    np.save(data / "cluster_ids.npy", cluster_ids)
    np.save(data / "centroids.npy", centroids)
    np.save(data / "category_centroids.npy", category_centroids, allow_pickle=True)

    with open(data / "paper_meta.jsonl", "w", encoding="utf-8") as fh:
        for paper in papers:
            fh.write(json.dumps(paper) + "\n")

    print(f"  Saved to {data_dir}/  ({time.time() - t5:.1f}s)")
    print(f"\nPipeline complete. Total time: {time.time() - t0:.1f}s")
    print(f"  embeddings: {embeddings.shape}  ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"  centroids:  {centroids.shape}")
    print(f"  categories: {len(category_centroids)}")
    print(f"  papers:     {len(papers)}")
