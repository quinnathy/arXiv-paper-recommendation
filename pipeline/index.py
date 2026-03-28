"""In-memory paper index for serving recommendations.

Loads all precomputed artifacts (embeddings, clusters, metadata) from disk
once at startup and holds them in memory for fast KNN lookups.
Designed to be cached via st.cache_resource.
"""

from __future__ import annotations

import numpy as np


class PaperIndex:
    """Holds all data needed for serving recommendations.

    Loaded once at Streamlit startup into memory via st.cache_resource.

    Attributes:
        embeddings: Shape (N, 768) float32 — all paper embeddings.
        cluster_ids: Shape (N,) int32 — cluster assignment per paper.
        centroids: Shape (k, 768) float32 — unit-norm cluster centroids.
        category_centroids: Dict mapping arXiv category string to unit-norm
            centroid vector (shape (768,) float32).
        paper_meta: Length-N list of dicts — paper metadata (id, title,
            abstract, categories, update_date, cluster_id).

    Memory note:
        At N=2M, embeddings.npy is ~5.9 GB.
        At N=50K (dev), it is ~147 MB.
    """

    def __init__(self, data_dir: str = "data") -> None:
        """Initialize the index with a data directory path.

        Args:
            data_dir: Path to the directory containing precomputed artifacts.

        Implementation:
            - Store data_dir.
            - Initialize all attributes to None.
        """
        # TODO: Set self.data_dir = data_dir
        # TODO: Set self.embeddings = None
        # TODO: Set self.cluster_ids = None
        # TODO: Set self.centroids = None
        # TODO: Set self.category_centroids = None
        # TODO: Set self.paper_meta = None
        raise NotImplementedError

    def load(self) -> None:
        """Load all artifacts from data_dir into memory.

        Reads the following files from self.data_dir:
            - embeddings.npy    → self.embeddings  (N, 768) float32
            - cluster_ids.npy   → self.cluster_ids (N,) int32
            - centroids.npy     → self.centroids   (k, 768) float32
            - category_centroids.npy → self.category_centroids dict
                                       (loaded with allow_pickle=True)
            - paper_meta.jsonl  → self.paper_meta  list[dict]

        After loading, verifies that len(embeddings) == len(paper_meta).
        Prints memory usage: embeddings.nbytes / 1e9 GB.
        """
        raise NotImplementedError

    def is_loaded(self) -> bool:
        """Check whether the index has been successfully loaded.

        Returns:
            True if self.embeddings is not None, False otherwise.
        """
        raise NotImplementedError
