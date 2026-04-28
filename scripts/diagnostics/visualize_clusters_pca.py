"""Generate an interactive PCA visualization from existing embeddings.

Scope:
    Standalone CLI wrapper for a quick 2D linear projection diagnostic. It
    samples rows from ``data/embeddings.npy`` and pairs them with metadata from
    ``data/paper_meta.jsonl`` and cluster IDs from ``data/cluster_ids.npy``.

Purpose:
    Build a lightweight embedding-space visualization colored by arXiv
    category, top-level category, or k-means cluster. PCA is useful for fast
    inspection, while UMAP is usually preferred for user-facing nonlinear
    latent-space views.

Artifacts produced:
    - data/diagnostics/pca_cluster_viz.html
    - data/diagnostics/pca_cluster_viz.png, when static export is available
    - data/diagnostics/pca_cluster_viz_coords.npy
    - data/diagnostics/pca_cluster_viz_indices.npy
    - data/diagnostics/pca_cluster_viz_metadata.json

Command:
    python scripts/diagnostics/visualize_clusters_pca.py \
      --sample-size 50000 \
      --color-by primary_category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.embedding_viz import generate_pca_visualization
from recommender.visualization import SUPPORTED_COLOR_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D PCA embedding visualization.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--diagnostics-dir", default="data/diagnostics")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--color-by", choices=SUPPORTED_COLOR_MODES, default="primary_category")
    parser.add_argument("--skip-png", action="store_true")
    args = parser.parse_args()

    payload = generate_pca_visualization(
        data_dir=args.data_dir,
        diagnostics_dir=args.diagnostics_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
        color_by=args.color_by,
        write_png=not args.skip_png,
    )
    print(f"Saved PCA visualization to {payload['paths']['html']}")
    print(f"Explained variance ratio: {payload['explained_variance_ratio']}")


if __name__ == "__main__":
    main()
