"""Run a MiniBatchKMeans K sweep over existing paper embeddings.

Scope:
    Standalone CLI wrapper for the diagnostic K sweep. It only reads existing
    offline artifacts, especially ``data/embeddings.npy``; it does not re-run
    SPECTER2 encoding and does not overwrite production cluster artifacts.

Purpose:
    Compare several MiniBatchKMeans K values as retrieval-index candidates.
    The output helps choose a K that balances exact-search cost, compactness,
    cluster size distribution, and stability. The automatic elbow value is a
    diagnostic hint, not an objective optimum.

Artifacts produced:
    - data/diagnostics/kmeans_k_sweep.csv
    - data/diagnostics/kmeans_k_sweep.json
    - data/diagnostics/kmeans_elbow.html
    - data/diagnostics/kmeans_elbow.png, when static export is available
    - data/diagnostics/kmeans_k_report.md
    - data/diagnostics/kmeans_k_sweep_sample_indices.npy, when sampling

Command:
    python scripts/diagnostics/diagnose_kmeans_k.py \
      --k-values 100 200 300 400 500 700 1000 \
      --sample-size 200000 \
      --random-state 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.kmeans import DEFAULT_K_VALUES, run_k_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose K choices for the k-means spatial index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--diagnostics-dir", default="data/diagnostics")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metric-sample-size", type=int, default=10_000)
    parser.add_argument("--skip-quality-metrics", action="store_true")
    args = parser.parse_args()

    payload = run_k_sweep(
        data_dir=args.data_dir,
        diagnostics_dir=args.diagnostics_dir,
        k_values=args.k_values,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
        metric_sample_size=args.metric_sample_size,
        compute_quality_metrics=not args.skip_quality_metrics,
    )
    print(f"Saved K sweep diagnostics to {payload['paths']['json']}")
    print(f"Estimated elbow K: {payload['estimated_elbow_k']}")
    print(f"Current production K: {payload['current_production_k']}")


if __name__ == "__main__":
    main()
