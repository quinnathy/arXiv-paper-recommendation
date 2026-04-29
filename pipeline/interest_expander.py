"""Free-text interest expansion and embedding.

Expands short user phrases into richer pseudo-queries suitable for SPECTER2,
then embeds them.  The expansion adds context tokens so the model can place
the vector in a meaningful region of the embedding space even when the raw
phrase is only a few words long.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Curated expansions for common short terms
# ---------------------------------------------------------------------------

FREE_TEXT_EXPANSIONS: dict[str, str] = {
    "healthcare": (
        "machine learning for healthcare, clinical prediction, electronic health "
        "records, medical imaging, biomedical data, diagnosis, treatment planning, "
        "and clinical decision support"
    ),
    "computational biology": (
        "computational biology, genomics, single-cell data, protein modeling, "
        "biological networks, perturbation modeling, and bioinformatics"
    ),
    "climate": (
        "climate modeling, weather forecasting, meteorological downscaling, "
        "earth system science, and climate data analysis"
    ),
    "nlp": (
        "natural language processing, text classification, information extraction, "
        "machine translation, question answering, and language understanding"
    ),
    "computer vision": (
        "computer vision, image classification, object detection, semantic "
        "segmentation, visual recognition, and image understanding"
    ),
    "robotics": (
        "robotics, robot learning, manipulation, grasping, navigation, "
        "locomotion, and embodied AI"
    ),
    "rl": (
        "reinforcement learning, policy optimization, multi-agent systems, "
        "offline RL, model-based RL, and decision making"
    ),
    "drug discovery": (
        "drug discovery, molecular generation, virtual screening, "
        "structure-based drug design, and molecular property prediction"
    ),
    "llm": (
        "large language models, pre-training, instruction tuning, "
        "reasoning, alignment, and retrieval-augmented generation"
    ),
    "llms": (
        "large language models, pre-training, instruction tuning, "
        "reasoning, alignment, and retrieval-augmented generation"
    ),
}

# Threshold: if the user already wrote something descriptive, skip expansion.
_MAX_SHORT_LEN = 80


def expand_interest(phrase: str) -> str:
    """Expand a short user phrase into a richer pseudo-query for embedding.

    Args:
        phrase: Raw user input, e.g. ``"healthcare"`` or
            ``"diffusion models for medical imaging"``.

    Returns:
        An expanded string suitable as a SPECTER2 paper title.
    """
    stripped = phrase.strip()
    if not stripped:
        return stripped

    # Already descriptive enough — return as-is.
    if len(stripped) > _MAX_SHORT_LEN:
        return stripped

    normalized = stripped.lower()
    if normalized in FREE_TEXT_EXPANSIONS:
        expanded = FREE_TEXT_EXPANSIONS[normalized]
    else:
        expanded = stripped

    return (
        f"Research papers about {expanded}. "
        f"This topic includes relevant methods, applications, datasets, "
        f"benchmarks, and recent scientific progress."
    )


def embed_free_text_interests(
    phrases: list[str],
    model: object,
) -> list[tuple[str, np.ndarray]]:
    """Expand and embed a list of user-provided free-text interest phrases.

    Args:
        phrases: Raw phrases from the onboarding text area.
        model: An :class:`~pipeline.embed.EmbeddingModel` instance.

    Returns:
        List of ``(original_phrase, embedding)`` pairs where each embedding
        is a unit-norm (768,) float32 vector.
    """
    if not phrases:
        return []

    expanded = [expand_interest(p) for p in phrases]
    papers = [{"title": text, "abstract": ""} for text in expanded]
    embs = model.embed_papers(papers)  # (n, 768), already L2-normalized

    return list(zip(phrases, embs))
