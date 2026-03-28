"""SPECTER2 embedding wrapper for arXiv papers.

Provides a thin interface around SentenceTransformer with the SPECTER2 model.
All embeddings are normalized to unit length so that dot product = cosine similarity.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Encodes paper titles+abstracts into 768-dim SPECTER2 embeddings.

    Usage:
        model = EmbeddingModel()
        embeddings = model.embed_papers([{"title": "...", "abstract": "..."}, ...])
    """

    MODEL_NAME = "allenai/specter2_base"

    def __init__(self) -> None:
        """Load the SentenceTransformer model.

        Sets device to "cuda" if a GPU is available, otherwise "cpu".
        """
        # TODO: Load SentenceTransformer(self.MODEL_NAME)
        # TODO: Set self.device based on torch.cuda.is_available()
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of raw text strings into normalized embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray of shape (len(texts), 768), dtype float32.
            Every row is unit-norm (normalize_embeddings=True).

        Implementation:
            - Call self.model.encode(texts, batch_size=64, normalize_embeddings=True)
            - Return the result as float32 ndarray.
        """
        raise NotImplementedError

    def embed_papers(self, papers: list[dict]) -> np.ndarray:
        """Encode paper dicts into normalized embeddings.

        Each paper dict must have "title" and "abstract" keys.
        The input text for each paper is "{title}. {abstract}".

        Args:
            papers: List of dicts, each with at least "title" and "abstract".

        Returns:
            np.ndarray of shape (len(papers), 768), dtype float32.

        Implementation:
            - Concatenate "{title}. {abstract}" for each paper.
            - Call self.embed_batch() on the concatenated texts.
            - Return the result.
        """
        raise NotImplementedError
