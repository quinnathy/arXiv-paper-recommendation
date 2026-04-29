"""Build joke-board embedding artifacts for the daily feed.

Writes:
    - data/joke_embeddings.npy
    - data/joke_embeddings_meta.json

Usage:
    python scripts/build_joke_embeddings.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.runtime import configure_single_thread_runtime

configure_single_thread_runtime()

from pipeline.embed import EmbeddingModel  # noqa: E402
from ui.domain_jokes import (  # noqa: E402
    JOKE_BOARDS,
    compute_joke_embeddings,
    save_joke_embedding_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build joke-board embedding artifacts for the daily feed."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where joke embedding artifacts will be written.",
    )
    args = parser.parse_args()

    model = EmbeddingModel()
    matrix = compute_joke_embeddings(model)
    save_joke_embedding_artifacts(
        matrix,
        data_dir=args.data_dir,
        metadata={
            "model_name": EmbeddingModel.MODEL_NAME,
            "adapter_name": EmbeddingModel.ADAPTER_NAME,
        },
    )
    print(
        f"Saved {len(JOKE_BOARDS)} joke-board embeddings to "
        f"{Path(args.data_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
