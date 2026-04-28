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
    type=int,
    default=500,
    help="Number of k-means clusters. Default 500.",
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
)
