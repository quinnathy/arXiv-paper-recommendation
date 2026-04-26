"""Build standalone concept embedding artifacts for onboarding.

This script intentionally does not touch paper embeddings, clusters, metadata,
or other offline pipeline outputs. It only writes:
    - data/concept_embeddings.npy
    - data/concept_embeddings_meta.json
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

from pipeline.concept_tags import (  # noqa: E402
    compute_concept_embeddings,
    save_concept_embedding_artifacts,
)
from pipeline.embed import EmbeddingModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build concept embedding artifacts for onboarding."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where concept embedding artifacts will be written.",
    )
    args = parser.parse_args()

    model = EmbeddingModel()
    embeddings = compute_concept_embeddings(model)
    save_concept_embedding_artifacts(
        embeddings,
        data_dir=args.data_dir,
        metadata={
            "model_name": EmbeddingModel.MODEL_NAME,
            "adapter_name": EmbeddingModel.ADAPTER_NAME,
        },
    )
    print(
        f"Saved {len(embeddings)} concept embeddings to "
        f"{Path(args.data_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
