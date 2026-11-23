"""Hybrid retrieval = Reciprocal Rank Fusion (RRF) of several rankings.

RRF (Cormack et al., 2009): score(d) = sum over rankings of 1 / (k + rank(d)).
It only uses ranks, so I don't need to normalise BM25 scores vs cosine scores
(they live on totally different scales).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of doc ids. Returns (doc_id, rrf_score), best first."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    # sort by score, then by doc_id so the order is deterministic when scores tie
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    """Runs several retrievers and fuses their top-`depth` lists with RRF."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 50) -> None:
        if not retrievers:
            raise ValueError("need at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        depth = max(self.depth, k)
        rankings = [[doc_id for doc_id, _ in r.search(query, depth)] for r in self.retrievers]
        return rrf_fuse(rankings, self.rrf_k)[:k]
