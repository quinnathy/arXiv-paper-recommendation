"""Retrain the MiniBatchKMeans retrieval index for a chosen K.

Scope:
    Standalone CLI wrapper for fitting the corpus-level k-means spatial index
    on the full existing embedding matrix. It reuses ``data/embeddings.npy``
    and does not re-embed papers.

Purpose:
    Produce a validated set of cluster IDs, unit-normalized centroids, cluster
    sizes, and metadata for one selected K after reviewing diagnostics. These
    clusters are anonymous retrieval buckets, not semantic topics.

Artifacts produced:
    For ``--output-prefix data/kmeans_k700``:
    - data/kmeans_k700_cluster_ids.npy
    - data/kmeans_k700_centroids.npy
    - data/kmeans_k700_cluster_sizes.npy
    - data/kmeans_k700_metadata.json

    With ``--make-current`` it also copies the chosen cluster IDs and centroids
    to the production artifact paths expected by the recommender:
    - data/cluster_ids.npy
    - data/centroids.npy
    - data/paper_meta.jsonl, updated so metadata cluster IDs stay aligned
    - data/paper_meta_before_kmeans_make_current.jsonl, first-run backup

Command:
    python scripts/diagnostics/retrain_kmeans_index.py \
      --k 700 \
      --batch-size 8192 \
      --max-iter 100 \
      --random-state 42 \
      --output-prefix data/kmeans_k700 \
      --make-current  # promote the current k and generate relevant artifacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.kmeans import retrain_kmeans_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the k-means spatial index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--make-current", action="store_true")
    args = parser.parse_args()

    metadata = retrain_kmeans_index(
        k=args.k,
        data_dir=args.data_dir,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
        make_current=args.make_current,
    )
    print(f"Saved retrained artifacts with prefix metadata at {metadata['paths']['metadata']}")
    if args.make_current:
        print("Copied selected artifacts to current production paths.")


if __name__ == "__main__":
    main()
