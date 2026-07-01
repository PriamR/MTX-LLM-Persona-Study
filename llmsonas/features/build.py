"""Per-user behavioural feature vectors.

Two blocks, concatenated:
  numeric   log1p(playtime / games / reviews) + recommend flag, z-scored;
  text      MiniLM embedding of the review, PCA-reduced and z-scored.

The same MiniLM model (all-MiniLM-L6-v2) the Core paper used. Tag exposure is
omitted here: appreviews is single-app, so tags would need Steam Web API
enrichment (deferred to the full run) — the smoke test proves the pipeline
connects on numeric + text signal. Datasets that ship no review text (the 2024
dump) use ``numeric_features`` and skip the text block entirely.
"""
from __future__ import annotations

import numpy as np

from llmsonas.data.ingest import UserRecord

EMBED_MODEL = "all-MiniLM-L6-v2"
_embedder = None


def _zscore(X: np.ndarray) -> np.ndarray:
    """Column z-score; zero-variance columns pass through unscaled."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def numeric_block(records: list[UserRecord]) -> np.ndarray:
    rows = [
        [
            np.log1p(r.playtime_forever),
            np.log1p(r.playtime_at_review),
            np.log1p(r.num_games_owned),
            np.log1p(r.num_reviews),
            1.0 if r.voted_up else 0.0,
        ]
        for r in records
    ]
    return np.asarray(rows, dtype=float)


def numeric_features(records: list[UserRecord]) -> np.ndarray:
    """Behavioural feature matrix from numeric signal alone (z-scored).

    Used when the source carries no review text (the 2024 dump): the text block
    is simply absent, so clustering and the homophily graph run on the
    behavioural numbers — playtime, tenure proxy, ownership, vocalness, stance.
    """
    return _zscore(numeric_block(records))


def build_features(
    records: list[UserRecord], *, embed_dims: int = 16, seed: int = 42
) -> np.ndarray:
    """Return the (n_users, d) feature matrix aligned to ``records`` order."""
    from sklearn.decomposition import PCA

    num = _zscore(numeric_block(records))

    emb = _get_embedder().encode(
        [r.review for r in records],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=float)

    k = min(embed_dims, emb.shape[1], max(2, len(records) - 1))
    emb = PCA(n_components=k, random_state=seed).fit_transform(emb)
    emb = _zscore(emb)

    return np.hstack([num, emb])
