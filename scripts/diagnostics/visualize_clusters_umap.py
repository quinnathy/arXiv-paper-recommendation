"""Generate an interactive UMAP visualization from existing embeddings.

Scope:
    Standalone CLI wrapper for a 2D nonlinear embedding-space projection. It
    samples rows from ``data/embeddings.npy`` and joins them with paper
    metadata and current k-means cluster IDs. It requires ``umap-learn``.

Purpose:
    Build the preferred latent-space visualization for app-facing exploration,
    with points colored by primary arXiv category, top-level category, or
    anonymous k-means retrieval cluster.

Artifacts produced:
    - data/diagnostics/umap_cluster_viz.html
    - data/diagnostics/umap_cluster_viz.png, when static export is available
    - data/diagnostics/umap_cluster_viz_coords.npy
    - data/diagnostics/umap_cluster_viz_indices.npy
    - data/diagnostics/umap_cluster_viz_model.pkl
    - data/diagnostics/umap_cluster_viz_metadata.json

Command:
    python scripts/diagnostics/visualize_clusters_umap.py \
      --sample-size 50000 \
      --color-by top_level_category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.embedding_viz import generate_umap_visualization
from recommender.visualization import SUPPORTED_COLOR_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D UMAP embedding visualization.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--diagnostics-dir", default="data/diagnostics")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--color-by", choices=SUPPORTED_COLOR_MODES, default="primary_category")
    parser.add_argument("--skip-png", action="store_true")
    args = parser.parse_args()

    try:
        payload = generate_umap_visualization(
            data_dir=args.data_dir,
            diagnostics_dir=args.diagnostics_dir,
            sample_size=args.sample_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
            color_by=args.color_by,
            write_png=not args.skip_png,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Saved UMAP visualization to {payload['paths']['html']}")


if __name__ == "__main__":
    main()
