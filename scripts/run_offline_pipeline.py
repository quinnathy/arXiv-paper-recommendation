"""CLI script to run the offline data pipeline.

Usage:
    python scripts/run_offline_pipeline.py               # Process all papers (~2M)
    python scripts/run_offline_pipeline.py --limit 50000  # Dev: ~5 min on CPU
    python scripts/run_offline_pipeline.py --limit 10000  # Quick test

Produces all artifacts in data/ that the app needs to serve recommendations.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(
    description="Run the ArXiv offline pipeline: download → embed → cluster → save."
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Max papers to process (None = all). Use 10000-50000 for dev.",
)
parser.add_argument(
    "--k",
    "--kmeans-k",
    dest="k",
    type=int,
    default=500,
    help="Number of k-means clusters. Default 500.",
)
parser.add_argument(
    "--kmeans-batch-size",
    type=int,
    default=4096,
    help="MiniBatchKMeans batch size for production clustering.",
)
parser.add_argument(
    "--kmeans-max-iter",
    type=int,
    default=100,
    help="MiniBatchKMeans max iterations for production clustering.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible sampling when --limit is used.",
)
parser.add_argument(
    "--categories",
    type=str,
    default="",
    help='Comma-separated category filters, e.g. "cs.LG,cs.CV".',
)
parser.add_argument(
    "--hf-endpoint",
    type=str,
    default="",
    help="Optional Hugging Face endpoint override (e.g. https://hf-mirror.com).",
)
parser.add_argument(
    "--disable-hf-transfer",
    action="store_true",
    help="Disable HF transfer acceleration (sets HF_HUB_ENABLE_HF_TRANSFER=0).",
)
parser.add_argument(
    "--run-kmeans-diagnostics",
    dest="run_kmeans_diagnostics",
    action="store_true",
    help="Run the K sweep diagnostics after generating embeddings and production clusters.",
)
parser.add_argument(
    "--skip-kmeans-diagnostics",
    dest="run_kmeans_diagnostics",
    action="store_false",
    help="Skip K sweep diagnostics.",
)
parser.set_defaults(run_kmeans_diagnostics=False)
parser.add_argument(
    "--kmeans-diagnostic-k-values",
    nargs="+",
    type=int,
    default=None,
    help="K grid for optional offline K sweep diagnostics.",
)
parser.add_argument(
    "--kmeans-diagnostic-sample-size",
    type=int,
    default=200000,
    help="Sample size for optional offline K sweep diagnostics.",
)
parser.add_argument(
    "--run-pca-viz",
    dest="run_pca_viz",
    action="store_true",
    help="Generate PCA visualization diagnostics after artifact save.",
)
parser.add_argument(
    "--skip-pca-viz",
    dest="run_pca_viz",
    action="store_false",
    help="Skip PCA visualization diagnostics.",
)
parser.set_defaults(run_pca_viz=False)
parser.add_argument(
    "--run-umap-viz",
    dest="run_umap_viz",
    action="store_true",
    help="Generate UMAP visualization diagnostics after artifact save.",
)
parser.add_argument(
    "--run-umap",
    dest="run_umap_viz",
    action="store_true",
    help="Alias for --run-umap-viz.",
)
parser.add_argument(
    "--skip-umap",
    dest="run_umap_viz",
    action="store_false",
    help="Skip UMAP visualization diagnostics.",
)
parser.set_defaults(run_umap_viz=False)
parser.add_argument(
    "--viz-sample-size",
    type=int,
    default=50000,
    help="Sample size for optional PCA/UMAP visualizations.",
)
parser.add_argument(
    "--viz-color-by",
    choices=["primary_category", "top_level_category", "cluster"],
    default="primary_category",
    help="Default color mode for generated visualization artifacts.",
)
args, _unknown = parser.parse_known_args()

categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
if args.hf_endpoint.strip():
    os.environ["HF_ENDPOINT"] = args.hf_endpoint.strip()
if args.disable_hf_transfer:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from pipeline.runtime import configure_single_thread_runtime

configure_single_thread_runtime()

from pipeline.offline import run

run(
    limit=args.limit,
    k=args.k,
    seed=args.seed,
    categories=categories,
    kmeans_batch_size=args.kmeans_batch_size,
    kmeans_max_iter=args.kmeans_max_iter,
    run_kmeans_diagnostics=args.run_kmeans_diagnostics,
    kmeans_diagnostic_k_values=args.kmeans_diagnostic_k_values,
    kmeans_diagnostic_sample_size=args.kmeans_diagnostic_sample_size,
    run_pca_viz=args.run_pca_viz,
    run_umap_viz=args.run_umap_viz,
    viz_sample_size=args.viz_sample_size,
    viz_color_by=args.viz_color_by,
)
