"""Reranking and diversity filtering for recommendation candidates.

Takes the raw KNN candidates and applies:
1. Recency boost — newer papers get a score bonus.
2. Diversity filter — ensures selected papers come from different clusters.
"""

from __future__ import annotations


def recency_score(published_date: str, halflife_days: float = 30.0) -> float:
    """Compute a recency bonus score for a paper based on its publication date.

    More recent papers get higher scores, decaying exponentially.

    Args:
        published_date: ISO format date string from paper_meta["update_date"].
        halflife_days: Controls how fast the recency score decays.
            Default 30.0 days.

    Returns:
        Float in (0, 1]. Recent papers → ~1.0, old papers → small positive.

    Implementation:
        - Parse published_date (ISO format).
        - age_days = (datetime.now() - published).days
        - Clamp age to max 365 days to avoid near-zero scores on old papers.
        - return exp(-age_days / halflife_days)
    """
    raise NotImplementedError


def rerank_and_select(
    candidates: list[tuple[float, dict]],
    recency_weight: float = 0.25,
    n: int = 3,
) -> list[dict]:
    """Rerank candidates by combined similarity + recency, then select diverse top-n.

    Args:
        candidates: List of (similarity_score, paper_meta_dict) tuples
            from the retrieval stage.
        recency_weight: Weight of the recency bonus in the final score.
            Final score = sim_score + recency_weight * recency_score(date).
        n: Number of papers to select.

    Returns:
        List of n paper_meta dicts, each with an added "rec_score" key
        containing the final combined score.

    Implementation:
        - For each (sim_score, meta) in candidates:
            - Compute final score = sim_score + recency_weight * recency_score(meta["update_date"])
            - Attach score to meta dict as meta["rec_score"]
        - Sort all candidates by final score descending.
        - Diversity pass: iterate through sorted candidates. Add a paper only
          if its cluster_id has not been used yet in the selection.
        - Stop when n papers are selected.
        - Return the list of n meta dicts.
    """
    raise NotImplementedError
